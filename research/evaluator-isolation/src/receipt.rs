//! Canonical, versioned evaluator receipts signed with Ed25519.
//!
//! Receipt bytes are a fixed little-endian binary schema with explicit domain
//! separation. Do not replace this encoding with ordinary JSON serialization:
//! receipt signatures bind the canonical bytes exactly.
//!
//! The private key is loaded only from the evaluator's systemd credential
//! directory. Candidate code must never receive this path or credential.
//! Ed25519 operations use ed25519-dalek; no cryptographic primitive is
//! implemented in this module.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use zeroize::Zeroize;

use super::{
    IntakeError, RESOLVE_BENEATH, RESOLVE_NO_MAGICLINKS, RESOLVE_NO_SYMLINKS,
    open_absolute_directory, openat2_beneath, single_component,
};

const DOMAIN: &[u8] = b"SYMTHAEA-EVALUATOR-RECEIPT-V1\0";
const SIGNATURE_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ReceiptDisposition {
    Completed = 1,
    Invalid = 2,
    TimedOut = 3,
    Abstained = 4,
    OutOfBudget = 5,
}

impl TryFrom<u8> for ReceiptDisposition {
    type Error = ReceiptError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Completed),
            2 => Ok(Self::Invalid),
            3 => Ok(Self::TimedOut),
            4 => Ok(Self::Abstained),
            5 => Ok(Self::OutOfBudget),
            _ => Err(ReceiptError::MalformedEnvelope),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptPayload {
    pub schema_version: u16,
    /// First 16 bytes of SHA-256(Ed25519 public key).
    pub signing_key_id: [u8; 16],
    pub run_id: [u8; 16],
    pub nonce: [u8; 16],
    pub monotonic_counter: u64,
    pub issued_at_unix_ns: u64,

    pub task_sha256: [u8; 32],
    /// Commitment digest only; never place oracle bytes in a public receipt.
    pub oracle_commitment_sha256: [u8; 32],
    pub candidate_prediction_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
    pub evaluator_profile_sha256: [u8; 32],

    pub total_exposure_us: u64,
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,

    pub bytes_used: u64,
    pub byte_budget: u64,
    pub retained_byte_us: u128,
    pub compute_used: u64,
    pub compute_budget: u64,
    pub refreshes_used: u64,
    pub refresh_budget: u64,
    pub observations_used: u64,
    pub observation_budget: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub measured_energy_nj: Option<u64>,
    pub energy_budget_nj: Option<u64>,

    pub disposition: ReceiptDisposition,
    /// Stable evaluator-defined numeric rejection/failure code; zero on success.
    pub failure_code: u16,
}

impl ReceiptPayload {
    pub fn validate(&self) -> Result<(), ReceiptError> {
        if self.schema_version != 1
            || self.run_id == [0; 16]
            || self.nonce == [0; 16]
            || self.monotonic_counter == 0
            || self.issued_at_unix_ns == 0
            || self.total_exposure_us == 0
        {
            return Err(ReceiptError::InvalidPayload);
        }
        let exposure_total = self
            .wrong_belief_us
            .checked_add(self.abstention_us)
            .and_then(|value| value.checked_add(self.missing_output_us))
            .ok_or(ReceiptError::ArithmeticOverflow)?;
        if exposure_total > self.total_exposure_us {
            return Err(ReceiptError::ExposureExceedsEpisode);
        }
        if self.energy_budget_nj.is_some() && self.measured_energy_nj.is_none() {
            return Err(ReceiptError::EnergyMeasurementMissing);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ReceiptError> {
        self.validate()?;
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(DOMAIN);
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        out.extend_from_slice(&self.signing_key_id);
        out.extend_from_slice(&self.run_id);
        out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.monotonic_counter.to_le_bytes());
        out.extend_from_slice(&self.issued_at_unix_ns.to_le_bytes());

        out.extend_from_slice(&self.task_sha256);
        out.extend_from_slice(&self.oracle_commitment_sha256);
        out.extend_from_slice(&self.candidate_prediction_sha256);
        out.extend_from_slice(&self.model_sha256);
        out.extend_from_slice(&self.config_sha256);
        out.extend_from_slice(&self.evaluator_profile_sha256);

        out.extend_from_slice(&self.total_exposure_us.to_le_bytes());
        out.extend_from_slice(&self.wrong_belief_us.to_le_bytes());
        out.extend_from_slice(&self.abstention_us.to_le_bytes());
        out.extend_from_slice(&self.missing_output_us.to_le_bytes());

        out.extend_from_slice(&self.bytes_used.to_le_bytes());
        out.extend_from_slice(&self.byte_budget.to_le_bytes());
        out.extend_from_slice(&self.retained_byte_us.to_le_bytes());
        out.extend_from_slice(&self.compute_used.to_le_bytes());
        out.extend_from_slice(&self.compute_budget.to_le_bytes());
        out.extend_from_slice(&self.refreshes_used.to_le_bytes());
        out.extend_from_slice(&self.refresh_budget.to_le_bytes());
        out.extend_from_slice(&self.observations_used.to_le_bytes());
        out.extend_from_slice(&self.observation_budget.to_le_bytes());
        out.extend_from_slice(&self.read_bytes.to_le_bytes());
        out.extend_from_slice(&self.write_bytes.to_le_bytes());

        encode_optional_u64(&mut out, self.measured_energy_nj);
        encode_optional_u64(&mut out, self.energy_budget_nj);
        out.push(self.disposition as u8);
        out.extend_from_slice(&self.failure_code.to_le_bytes());
        Ok(out)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedReceipt {
    pub payload: ReceiptPayload,
    pub signature: [u8; SIGNATURE_BYTES],
}

impl SignedReceipt {
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReceiptError> {
        let mut out = self.payload.canonical_bytes()?;
        out.extend_from_slice(&self.signature);
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReceiptError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect(DOMAIN)?;
        let schema_version = u16::from_le_bytes(decoder.take::<2>()?);
        let signing_key_id = decoder.take::<16>()?;
        let run_id = decoder.take::<16>()?;
        let nonce = decoder.take::<16>()?;
        let monotonic_counter = u64::from_le_bytes(decoder.take::<8>()?);
        let issued_at_unix_ns = u64::from_le_bytes(decoder.take::<8>()?);

        let task_sha256 = decoder.take::<32>()?;
        let oracle_commitment_sha256 = decoder.take::<32>()?;
        let candidate_prediction_sha256 = decoder.take::<32>()?;
        let model_sha256 = decoder.take::<32>()?;
        let config_sha256 = decoder.take::<32>()?;
        let evaluator_profile_sha256 = decoder.take::<32>()?;

        let total_exposure_us = u64::from_le_bytes(decoder.take::<8>()?);
        let wrong_belief_us = u64::from_le_bytes(decoder.take::<8>()?);
        let abstention_us = u64::from_le_bytes(decoder.take::<8>()?);
        let missing_output_us = u64::from_le_bytes(decoder.take::<8>()?);

        let bytes_used = u64::from_le_bytes(decoder.take::<8>()?);
        let byte_budget = u64::from_le_bytes(decoder.take::<8>()?);
        let retained_byte_us = u128::from_le_bytes(decoder.take::<16>()?);
        let compute_used = u64::from_le_bytes(decoder.take::<8>()?);
        let compute_budget = u64::from_le_bytes(decoder.take::<8>()?);
        let refreshes_used = u64::from_le_bytes(decoder.take::<8>()?);
        let refresh_budget = u64::from_le_bytes(decoder.take::<8>()?);
        let observations_used = u64::from_le_bytes(decoder.take::<8>()?);
        let observation_budget = u64::from_le_bytes(decoder.take::<8>()?);
        let read_bytes = u64::from_le_bytes(decoder.take::<8>()?);
        let write_bytes = u64::from_le_bytes(decoder.take::<8>()?);

        let measured_energy_nj = decoder.optional_u64()?;
        let energy_budget_nj = decoder.optional_u64()?;
        let disposition = ReceiptDisposition::try_from(decoder.take::<1>()?[0])?;
        let failure_code = u16::from_le_bytes(decoder.take::<2>()?);
        let signature = decoder.take::<SIGNATURE_BYTES>()?;
        if !decoder.is_empty() {
            return Err(ReceiptError::MalformedEnvelope);
        }

        let payload = ReceiptPayload {
            schema_version,
            signing_key_id,
            run_id,
            nonce,
            monotonic_counter,
            issued_at_unix_ns,
            task_sha256,
            oracle_commitment_sha256,
            candidate_prediction_sha256,
            model_sha256,
            config_sha256,
            evaluator_profile_sha256,
            total_exposure_us,
            wrong_belief_us,
            abstention_us,
            missing_output_us,
            bytes_used,
            byte_budget,
            retained_byte_us,
            compute_used,
            compute_budget,
            refreshes_used,
            refresh_budget,
            observations_used,
            observation_budget,
            read_bytes,
            write_bytes,
            measured_energy_nj,
            energy_budget_nj,
            disposition,
            failure_code,
        };
        payload.validate()?;
        Ok(Self { payload, signature })
    }

    pub fn envelope_sha256(&self) -> Result<[u8; 32], ReceiptError> {
        Ok(Sha256::digest(self.to_bytes()?).into())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiptError {
    InvalidPayload,
    ExposureExceedsEpisode,
    EnergyMeasurementMissing,
    ArithmeticOverflow,
    MalformedEnvelope,
    SigningKeyIdMismatch,
    SignatureInvalid,
    CredentialMalformed,
    CredentialIntake(IntakeError),
}

impl fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPayload => write!(f, "receipt payload has invalid version, run ID, nonce, counter or timestamp"),
            Self::ExposureExceedsEpisode => write!(f, "receipt exposure exceeds total episode duration"),
            Self::EnergyMeasurementMissing => write!(f, "receipt declares an energy budget but contains no measured energy"),
            Self::ArithmeticOverflow => write!(f, "receipt arithmetic overflowed"),
            Self::MalformedEnvelope => write!(f, "receipt envelope is malformed or has trailing bytes"),
            Self::SigningKeyIdMismatch => write!(f, "receipt signing key ID does not match the trusted public key"),
            Self::SignatureInvalid => write!(f, "receipt Ed25519 signature is invalid"),
            Self::CredentialMalformed => write!(f, "systemd signing credential must be a 32-byte regular file with one link"),
            Self::CredentialIntake(error) => write!(f, "cannot securely open evaluator signing credential: {error}"),
        }
    }
}

impl std::error::Error for ReceiptError {}

/// Sign a receipt after the evaluator has finalized its immutable measurement
/// record. The key identifier is bound into the canonical signed payload.
pub fn sign_receipt(
    mut payload: ReceiptPayload,
    signing_key: &SigningKey,
) -> Result<SignedReceipt, ReceiptError> {
    let verifying_key = signing_key.verifying_key();
    payload.signing_key_id = key_id(&verifying_key);
    let message = payload.canonical_bytes()?;
    let signature = signing_key.sign(&message).to_bytes();
    Ok(SignedReceipt { payload, signature })
}

/// Verify with an externally trusted, allow-listed Ed25519 public key.
pub fn verify_receipt(
    receipt: &SignedReceipt,
    verifying_key: &VerifyingKey,
) -> Result<(), ReceiptError> {
    if receipt.payload.signing_key_id != key_id(verifying_key) {
        return Err(ReceiptError::SigningKeyIdMismatch);
    }
    let message = receipt.payload.canonical_bytes()?;
    let signature = Signature::from_bytes(&receipt.signature);
    verifying_key
        .verify_strict(&message, &signature)
        .map_err(|_| ReceiptError::SignatureInvalid)
}

pub fn decode_and_verify(
    bytes: &[u8],
    verifying_key: &VerifyingKey,
) -> Result<SignedReceipt, ReceiptError> {
    let receipt = SignedReceipt::from_bytes(bytes)?;
    verify_receipt(&receipt, verifying_key)?;
    Ok(receipt)
}

/// Read a raw 32-byte Ed25519 seed from the systemd credential directory.
///
/// The unit must set LoadCredentialEncrypted=receipt-signing-key:... so this
/// directory and file are visible only to the evaluator UID. The filename is
/// fixed, no symlink is followed, and the file must have one hard link.
pub fn load_signing_key_from_systemd_credential(
    credential_directory: &Path,
) -> Result<SigningKey, ReceiptError> {
    let directory = open_absolute_directory(credential_directory)
        .map_err(ReceiptError::CredentialIntake)?;
    let name = single_component("receipt-signing-key")
        .map_err(ReceiptError::CredentialIntake)?;
    let fd = openat2_beneath(
        directory.as_raw_fd(),
        &name,
        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        0,
        RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    )
    .map_err(ReceiptError::CredentialIntake)?;
    // SAFETY: openat2 returned a new owned credential descriptor.
    let mut file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata().map_err(|_| ReceiptError::CredentialMalformed)?;
    if !metadata.file_type().is_file()
        || metadata.len() != 32
        || metadata.nlink() != 1
        || metadata.mode() & 0o077 != 0
    {
        return Err(ReceiptError::CredentialMalformed);
    }
    let mut seed = [0_u8; 32];
    let read_result = (|| -> Result<(), std::io::Error> {
        file.read_exact(&mut seed)?;
        let mut extra = [0_u8; 1];
        if file.read(&mut extra)? != 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "credential is longer than 32 bytes"));
        }
        Ok(())
    })();
    if read_result.is_err() {
        seed.zeroize();
        return Err(ReceiptError::CredentialMalformed);
    }
    let signing_key = SigningKey::from_bytes(&seed);
    seed.fill(0);
    Ok(signing_key)
}

/// Persistent evaluator-local replay ledger.
///
/// The ledger directory must be pre-created with mode 0700 and owned by the
/// evaluator UID. It is intentionally not shared with the candidate. A
/// receipt is consumed only after its Ed25519 signature verifies. Unique
/// create-only markers bind nonce, run ID, and per-key monotonic counter;
/// a flock-protected high-water file serializes concurrent evaluator workers.
///
/// If a crash occurs after marker creation but before counter update, the
/// marker remains consumed. This can waste a run ID/counter but fails closed.
pub struct ReplayLedger {
    directory: PathBuf,
}

impl ReplayLedger {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self { directory: directory.into() }
    }

    /// Verify the signature, then atomically consume the run/nonce/counter.
    /// The returned digest is the SHA-256 of the signed canonical envelope.
    pub fn verify_and_consume(
        &self,
        receipt: &SignedReceipt,
        verifying_key: &VerifyingKey,
    ) -> Result<[u8; 32], ReplayError> {
        verify_receipt(receipt, verifying_key).map_err(ReplayError::Receipt)?;
        let payload = &receipt.payload;
        let envelope_digest = receipt
            .envelope_sha256()
            .map_err(ReplayError::Receipt)?;

        let directory = open_absolute_directory(&self.directory)
            .map_err(ReplayError::Intake)?;
        let directory_metadata = directory.metadata().map_err(|error| ReplayError::Io(error.kind()))?;
        if !directory_metadata.file_type().is_dir()
            || directory_metadata.uid() != unsafe { libc::geteuid() }
            || directory_metadata.mode() & 0o077 != 0
        {
            return Err(ReplayError::LedgerDirectoryNotPrivate);
        }
        let key = key_id(verifying_key);
        let counter_name = single_component(&format!("counter-{}.bin", hex(&key)))
            .map_err(ReplayError::Intake)?;
        let counter_fd = openat2_beneath(
            directory.as_raw_fd(),
            &counter_name,
            libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
            RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
        )
        .map_err(ReplayError::Intake)?;
        // SAFETY: openat2 returned a new owned file descriptor.
        let mut counter_file = unsafe { File::from_raw_fd(counter_fd) };
        let counter_metadata = counter_file.metadata().map_err(ReplayError::Io)?;
        if !counter_metadata.file_type().is_file()
            || counter_metadata.nlink() != 1
            || counter_metadata.uid() != unsafe { libc::geteuid() }
            || counter_metadata.mode() & 0o077 != 0
            || counter_metadata.len() > 8
        {
            return Err(ReplayError::CounterLedgerCorrupt);
        }

        // SAFETY: flock is applied to the live counter-file descriptor. The
        // lock is released when the descriptor is closed on every return path.
        let lock_result = unsafe { libc::flock(counter_file.as_raw_fd(), libc::LOCK_EX) };
        if lock_result != 0 {
            return Err(ReplayError::Io(std::io::Error::last_os_error().kind()));
        }

        counter_file.seek(SeekFrom::Start(0)).map_err(ReplayError::Io)?;
        let mut counter_bytes = Vec::new();
        counter_file.read_to_end(&mut counter_bytes).map_err(ReplayError::Io)?;
        let previous_counter = match counter_bytes.len() {
            0 => 0,
            8 => u64::from_le_bytes(
                counter_bytes.as_slice().try_into().map_err(|_| ReplayError::CounterLedgerCorrupt)?,
            ),
            _ => return Err(ReplayError::CounterLedgerCorrupt),
        };
        if payload.monotonic_counter <= previous_counter {
            return Err(ReplayError::CounterNotMonotonic {
                previous: previous_counter,
                supplied: payload.monotonic_counter,
            });
        }

        let nonce_name = single_component(&format!("nonce-{}", hex(&payload.nonce)))
            .map_err(ReplayError::Intake)?;
        let run_name = single_component(&format!("run-{}", hex(&payload.run_id)))
            .map_err(ReplayError::Intake)?;
        let counter_marker = single_component(&format!(
            "used-{}-{:020}",
            hex(&key),
            payload.monotonic_counter
        ))
        .map_err(ReplayError::Intake)?;

        create_replay_marker(directory.as_raw_fd(), &nonce_name, &envelope_digest)?;
        create_replay_marker(directory.as_raw_fd(), &run_name, &envelope_digest)?;
        create_replay_marker(directory.as_raw_fd(), &counter_marker, &envelope_digest)?;

        // Persist create-only markers before updating the counter. A crash in
        // between leaves the run/nonce/counter consumed, never reusable.
        sync_replay_directory(&self.directory)?;

        counter_file.seek(SeekFrom::Start(0)).map_err(ReplayError::Io)?;
        counter_file.set_len(0).map_err(ReplayError::Io)?;
        counter_file
            .write_all(&payload.monotonic_counter.to_le_bytes())
            .map_err(ReplayError::Io)?;
        counter_file.sync_all().map_err(ReplayError::Io)?;
        sync_replay_directory(&self.directory)?;

        Ok(envelope_digest)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    Receipt(ReceiptError),
    Intake(IntakeError),
    Io(std::io::ErrorKind),
    CounterLedgerCorrupt,
    LedgerDirectoryNotPrivate,
    CounterNotMonotonic { previous: u64, supplied: u64 },
    AlreadyConsumed,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Receipt(error) => write!(f, "receipt verification failed: {error}"),
            Self::Intake(error) => write!(f, "replay ledger path operation failed: {error}"),
            Self::Io(kind) => write!(f, "replay ledger I/O failed: {kind:?}"),
            Self::CounterLedgerCorrupt => write!(f, "replay counter ledger is malformed or not private"),
            Self::LedgerDirectoryNotPrivate => write!(f, "replay ledger directory must be evaluator-owned and not accessible to group or other users"),
            Self::CounterNotMonotonic { previous, supplied } => write!(
                f,
                "receipt counter {supplied} is not greater than persisted counter {previous}"
            ),
            Self::AlreadyConsumed => write!(f, "receipt nonce, run ID or counter was already consumed"),
        }
    }
}

impl std::error::Error for ReplayError {}

fn create_replay_marker(
    directory_fd: i32,
    name: &CString,
    digest: &[u8; 32],
) -> Result<(), ReplayError> {
    let fd = match openat2_beneath(
        directory_fd,
        name,
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        0o600,
        RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    ) {
        Ok(fd) => fd,
        Err(IntakeError::OpenAt2(std::io::ErrorKind::AlreadyExists)) => {
            return Err(ReplayError::AlreadyConsumed);
        }
        Err(error) => return Err(ReplayError::Intake(error)),
    };
    // SAFETY: openat2 returned a new owned file descriptor.
    let mut marker = unsafe { File::from_raw_fd(fd) };
    marker.write_all(digest).map_err(|error| ReplayError::Io(error.kind()))?;
    marker.sync_all().map_err(|error| ReplayError::Io(error.kind()))?;
    Ok(())
}

fn sync_replay_directory(directory: &Path) -> Result<(), ReplayError> {
    let raw = CString::new(directory.as_os_str().as_bytes())
        .map_err(|_| ReplayError::Intake(IntakeError::InvalidDirectoryPath))?;
    let fd = openat2_raw(
        libc::AT_FDCWD,
        &raw,
        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        0,
        RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    )
    .map_err(ReplayError::Intake)?;
    // SAFETY: openat2 returned a new owned directory descriptor.
    let directory_file = unsafe { File::from_raw_fd(fd) };
    directory_file.sync_all().map_err(|error| ReplayError::Io(error.kind()))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn key_id(key: &VerifyingKey) -> [u8; 16] {
    let digest = Sha256::digest(key.to_bytes());
    let mut id = [0_u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

fn encode_optional_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.to_le_bytes());
        }
        None => {
            out.push(0);
            out.extend_from_slice(&0_u64.to_le_bytes());
        }
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], ReceiptError> {
        let end = self.cursor.checked_add(N).ok_or(ReceiptError::ArithmeticOverflow)?;
        let source = self.bytes.get(self.cursor..end).ok_or(ReceiptError::MalformedEnvelope)?;
        let mut result = [0_u8; N];
        result.copy_from_slice(source);
        self.cursor = end;
        Ok(result)
    }

    fn expect(&mut self, expected: &[u8]) -> Result<(), ReceiptError> {
        let end = self.cursor.checked_add(expected.len()).ok_or(ReceiptError::ArithmeticOverflow)?;
        let actual = self.bytes.get(self.cursor..end).ok_or(ReceiptError::MalformedEnvelope)?;
        if actual != expected {
            return Err(ReceiptError::MalformedEnvelope);
        }
        self.cursor = end;
        Ok(())
    }

    fn optional_u64(&mut self) -> Result<Option<u64>, ReceiptError> {
        let present = self.take::<1>()?[0];
        let value = u64::from_le_bytes(self.take::<8>()?);
        match present {
            0 if value == 0 => Ok(None),
            1 => Ok(Some(value)),
            _ => Err(ReceiptError::MalformedEnvelope),
        }
    }

    fn is_empty(&self) -> bool {
        self.cursor == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload() -> ReceiptPayload {
        ReceiptPayload {
            schema_version: 1,
            signing_key_id: [0; 16],
            run_id: [1; 16],
            nonce: [2; 16],
            monotonic_counter: 1,
            issued_at_unix_ns: 1_800_000_000_000_000_000,
            task_sha256: [3; 32],
            oracle_commitment_sha256: [4; 32],
            candidate_prediction_sha256: [5; 32],
            model_sha256: [6; 32],
            config_sha256: [7; 32],
            evaluator_profile_sha256: [8; 32],
            total_exposure_us: 100,
            wrong_belief_us: 20,
            abstention_us: 5,
            missing_output_us: 10,
            bytes_used: 1_024,
            byte_budget: 2_048,
            retained_byte_us: 10_240,
            compute_used: 100,
            compute_budget: 200,
            refreshes_used: 4,
            refresh_budget: 8,
            observations_used: 10,
            observation_budget: 20,
            read_bytes: 8_192,
            write_bytes: 1_024,
            measured_energy_nj: Some(50_000),
            energy_budget_nj: Some(100_000),
            disposition: ReceiptDisposition::Completed,
            failure_code: 0,
        }
    }

    #[test]
    fn signed_receipt_round_trips_and_verifies_strictly() {
        let signing_key = SigningKey::from_bytes(&[42_u8; 32]);
        let verifier = signing_key.verifying_key();
        let receipt = sign_receipt(payload(), &signing_key).unwrap();
        verify_receipt(&receipt, &verifier).unwrap();
        let bytes = receipt.to_bytes().unwrap();
        let decoded = decode_and_verify(&bytes, &verifier).unwrap();
        assert_eq!(decoded, receipt);
        assert_ne!(receipt.envelope_sha256().unwrap(), [0; 32]);
    }

    #[test]
    fn payload_mutation_invalidates_signature() {
        let signing_key = SigningKey::from_bytes(&[43_u8; 32]);
        let verifier = signing_key.verifying_key();
        let receipt = sign_receipt(payload(), &signing_key).unwrap();
        let mut changed = receipt.clone();
        changed.payload.candidate_prediction_sha256[0] ^= 1;
        assert_eq!(verify_receipt(&changed, &verifier), Err(ReceiptError::SignatureInvalid));
    }

    #[test]
    fn wrong_key_and_wrong_key_id_are_rejected() {
        let signing_key = SigningKey::from_bytes(&[44_u8; 32]);
        let other_key = SigningKey::from_bytes(&[45_u8; 32]);
        let receipt = sign_receipt(payload(), &signing_key).unwrap();
        assert_eq!(
            verify_receipt(&receipt, &other_key.verifying_key()),
            Err(ReceiptError::SigningKeyIdMismatch)
        );
    }

    #[test]
    fn malformed_and_trailing_envelope_bytes_are_rejected() {
        let signing_key = SigningKey::from_bytes(&[46_u8; 32]);
        let verifier = signing_key.verifying_key();
        let receipt = sign_receipt(payload(), &signing_key).unwrap();
        let mut bytes = receipt.to_bytes().unwrap();
        bytes.push(0);
        assert_eq!(
            SignedReceipt::from_bytes(&bytes),
            Err(ReceiptError::MalformedEnvelope)
        );
        bytes.pop();
        bytes[0] ^= 1;
        assert_eq!(
            SignedReceipt::from_bytes(&bytes),
            Err(ReceiptError::MalformedEnvelope)
        );
        assert!(decode_and_verify(&bytes, &verifier).is_err());
    }

    #[test]
    fn exposure_conservation_and_energy_measurement_are_enforced() {
        let mut invalid = payload();
        invalid.missing_output_us = 90;
        assert_eq!(invalid.validate(), Err(ReceiptError::ExposureExceedsEpisode));

        let mut no_energy = payload();
        no_energy.measured_energy_nj = None;
        assert_eq!(no_energy.validate(), Err(ReceiptError::EnergyMeasurementMissing));
    }

    #[test]
    fn replay_ledger_consumes_receipt_once_and_checks_monotonic_counter() {
        let temp = tempfile::tempdir().unwrap();
        let ledger_dir = temp.path().join("ledger");
        std::fs::create_dir(&ledger_dir).unwrap();
        let signing_key = SigningKey::from_bytes(&[51_u8; 32]);
        let verifier = signing_key.verifying_key();
        let ledger = ReplayLedger::new(&ledger_dir);

        let receipt = sign_receipt(payload(), &signing_key).unwrap();
        let digest = ledger.verify_and_consume(&receipt, &verifier).unwrap();
        assert_eq!(digest, receipt.envelope_sha256().unwrap());
        assert_eq!(
            ledger.verify_and_consume(&receipt, &verifier),
            Err(ReplayError::CounterNotMonotonic { previous: 1, supplied: 1 })
        );
    }

    #[test]
    fn replay_ledger_rejects_nonce_reuse_even_with_a_higher_counter() {
        let temp = tempfile::tempdir().unwrap();
        let ledger_dir = temp.path().join("ledger");
        std::fs::create_dir(&ledger_dir).unwrap();
        let signing_key = SigningKey::from_bytes(&[52_u8; 32]);
        let verifier = signing_key.verifying_key();
        let ledger = ReplayLedger::new(&ledger_dir);

        let first = sign_receipt(payload(), &signing_key).unwrap();
        ledger.verify_and_consume(&first, &verifier).unwrap();

        let mut second_payload = payload();
        second_payload.monotonic_counter = 2;
        second_payload.run_id = [9; 16];
        let second = sign_receipt(second_payload, &signing_key).unwrap();
        assert_eq!(
            ledger.verify_and_consume(&second, &verifier),
            Err(ReplayError::AlreadyConsumed)
        );
    }

    #[test]
    fn replay_ledger_rejects_counter_regression_and_run_reuse() {
        let temp = tempfile::tempdir().unwrap();
        let ledger_dir = temp.path().join("ledger");
        std::fs::create_dir(&ledger_dir).unwrap();
        let signing_key = SigningKey::from_bytes(&[53_u8; 32]);
        let verifier = signing_key.verifying_key();
        let ledger = ReplayLedger::new(&ledger_dir);

        let first = sign_receipt(payload(), &signing_key).unwrap();
        ledger.verify_and_consume(&first, &verifier).unwrap();

        let mut regression_payload = payload();
        regression_payload.monotonic_counter = 3;
        regression_payload.run_id = [8; 16];
        regression_payload.nonce = [7; 16];
        let regression = sign_receipt(regression_payload, &signing_key).unwrap();
        ledger.verify_and_consume(&regression, &verifier).unwrap();

        let mut reused_run_payload = payload();
        reused_run_payload.monotonic_counter = 4;
        reused_run_payload.nonce = [6; 16];
        let reused_run = sign_receipt(reused_run_payload, &signing_key).unwrap();
        assert_eq!(
            ledger.verify_and_consume(&reused_run, &verifier),
            Err(ReplayError::AlreadyConsumed)
        );
    }

    #[test]
    fn replay_ledger_requires_evaluator_private_directory() {
        let temp = tempfile::tempdir().unwrap();
        let ledger_dir = temp.path().join("ledger");
        std::fs::create_dir(&ledger_dir).unwrap();
        std::fs::set_permissions(
            &ledger_dir,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        ).unwrap();
        let signing_key = SigningKey::from_bytes(&[56_u8; 32]);
        let receipt = sign_receipt(payload(), &signing_key).unwrap();
        assert_eq!(
            ReplayLedger::new(&ledger_dir).verify_and_consume(&receipt, &signing_key.verifying_key()),
            Err(ReplayError::LedgerDirectoryNotPrivate)
        );
    }

    #[test]
    fn replay_ledger_rejects_counter_regression() {
        let temp = tempfile::tempdir().unwrap();
        let ledger_dir = temp.path().join("ledger");
        std::fs::create_dir(&ledger_dir).unwrap();
        let signing_key = SigningKey::from_bytes(&[57_u8; 32]);
        let verifier = signing_key.verifying_key();
        let ledger = ReplayLedger::new(&ledger_dir);

        let mut first_payload = payload();
        first_payload.monotonic_counter = 2;
        let first = sign_receipt(first_payload, &signing_key).unwrap();
        ledger.verify_and_consume(&first, &verifier).unwrap();

        let mut older_payload = payload();
        older_payload.monotonic_counter = 1;
        older_payload.run_id = [8; 16];
        older_payload.nonce = [9; 16];
        let older = sign_receipt(older_payload, &signing_key).unwrap();
        assert_eq!(
            ledger.verify_and_consume(&older, &verifier),
            Err(ReplayError::CounterNotMonotonic { previous: 2, supplied: 1 })
        );
    }

    #[test]
    fn invalid_signature_is_not_written_to_replay_ledger() {
        let temp = tempfile::tempdir().unwrap();
        let ledger_dir = temp.path().join("ledger");
        std::fs::create_dir(&ledger_dir).unwrap();
        let signing_key = SigningKey::from_bytes(&[54_u8; 32]);
        let other_key = SigningKey::from_bytes(&[55_u8; 32]);
        let receipt = sign_receipt(payload(), &signing_key).unwrap();
        let ledger = ReplayLedger::new(&ledger_dir);
        assert!(matches!(
            ledger.verify_and_consume(&receipt, &other_key.verifying_key()),
            Err(ReplayError::Receipt(ReceiptError::SigningKeyIdMismatch))
        ));
        assert_eq!(std::fs::read_dir(&ledger_dir).unwrap().count(), 0);
    }

    #[test]
    fn invalid_run_nonce_or_counter_cannot_be_signed() {
        let signing_key = SigningKey::from_bytes(&[47_u8; 32]);
        let mut invalid = payload();
        invalid.run_id = [0; 16];
        assert_eq!(
            sign_receipt(invalid, &signing_key),
            Err(ReceiptError::InvalidPayload)
        );
    }
}
