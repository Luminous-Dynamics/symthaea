// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-A: crash-durable, cross-process serialized enrollment allocation.
//!
//! P1ENR-A owns deterministic enrollment semantics. This module adds the local
//! runtime theorem: under one process-local mutex and one kernel file lock,
//! re-read one exact allocation head, derive exactly one next slot, atomically
//! persist the successor, fsync it, and read it back before acknowledging it.
//!
//! This is deliberately not anti-rollback or independent-witness evidence. A
//! privileged storage owner remains inside this failure domain. P1EIR must
//! independently witness the exact durable allocation before P1EACR may issue
//! scored participant authority.

use crate::evidence_digest::{
    canonical_json_bytes, canonical_json_sha256,
    perceptual_enrollment_lifecycle::{
        FrozenPerceptualEligibilityGateReceiptV1, FrozenPerceptualEnrollmentAllocationLedgerV1,
        FrozenPerceptualEnrollmentPolicyV1, PerceptualEnrollmentAllocationReceiptV1,
        PerceptualEnrollmentLifecycleIssueV1, allocate_next_enrollment_slot,
        validate_enrollment_allocation_ledger,
    },
    perceptual_participant_identity::{
        FrozenParticipantIdentityBoundaryPolicyV1, FrozenParticipantScheduleProjectionV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1,
    },
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleBookV1,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const DURABLE_ENROLLMENT_ALLOCATION_STATE_VERSION: &str =
    "mel003-durable-enrollment-allocation-state-v1";
const STATE_FILE_NAME: &str = "mel003-enrollment-allocation.state.json";
const LOCK_FILE_NAME: &str = ".mel003-enrollment-allocation.lock";
const MAX_STATE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableEnrollmentAllocationStateV1 {
    pub state_version: String,
    pub enrollment_policy_sha256: String,
    pub token_generation_receipt_sha256: String,
    pub participant_schedule_sha256: String,
    /// One gate per allocation in exact append order. These receipts contain
    /// study eligibility facts and opaque commitments, never raw recruitment identity.
    pub eligibility_gates: Vec<FrozenPerceptualEligibilityGateReceiptV1>,
    pub ledger: FrozenPerceptualEnrollmentAllocationLedgerV1,
    /// Local integrity/correlation digest only; not a privileged-storage
    /// anti-rollback claim.
    pub state_sha256: String,
}

/// Runtime-only local successor. The name is intentionally load-bearing:
/// this value has not yet crossed the independent P1EIR witness boundary and
/// must never be accepted directly as scored-collection authority.
///
/// It is also deliberately not Serialize/Deserialize because the participant
/// projection contains that participant's raw pseudonym. Persistable/witnessable
/// evidence remains the allocation receipt and exact ledger/state commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnwitnessedDurableEnrollmentAllocationV1 {
    pub previous_ledger_sha256: String,
    pub current_ledger_sha256: String,
    pub current_state_sha256: String,
    pub allocation: PerceptualEnrollmentAllocationReceiptV1,
    pub participant_schedule_projection: FrozenParticipantScheduleProjectionV1,
}

#[derive(Debug)]
pub enum DurableEnrollmentAllocationErrorV1 {
    UnsupportedPlatform,
    RootUnavailable,
    StateAlreadyExists,
    StateMissing,
    StateTooLarge,
    StateMalformed,
    StateIdentityMismatch,
    StateDigestMismatch,
    GateAllocationCardinalityMismatch { gates: usize, allocations: usize },
    GateAllocationOrderMismatch { index: usize },
    ExpectedLedgerHeadMismatch { expected: String, found: String },
    Lifecycle(Vec<PerceptualEnrollmentLifecycleIssueV1>),
    MissingSuccessorAllocation,
    ReadBackMismatch,
    Serialization,
    EntropyUnavailable,
    LocalLockPoisoned,
    KernelLockUnavailable,
    Io(std::io::Error),
}

impl std::fmt::Display for DurableEnrollmentAllocationErrorV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                write!(formatter, "durable enrollment allocation requires Linux")
            }
            Self::RootUnavailable => {
                write!(formatter, "durable enrollment allocation root is unavailable")
            }
            Self::StateAlreadyExists => {
                write!(formatter, "durable enrollment allocation state already exists")
            }
            Self::StateMissing => write!(formatter, "durable enrollment allocation state is missing"),
            Self::StateTooLarge => {
                write!(formatter, "durable enrollment allocation state exceeds its bound")
            }
            Self::StateMalformed => write!(formatter, "durable enrollment allocation state is malformed"),
            Self::StateIdentityMismatch => write!(
                formatter,
                "durable enrollment allocation state identity does not match frozen inputs"
            ),
            Self::StateDigestMismatch => {
                write!(formatter, "durable enrollment allocation state digest mismatch")
            }
            Self::GateAllocationCardinalityMismatch { gates, allocations } => write!(
                formatter,
                "eligibility-gate/allocation cardinality mismatch: {gates} gates for {allocations} allocations"
            ),
            Self::GateAllocationOrderMismatch { index } => write!(
                formatter,
                "eligibility gate does not match allocation at index {index}"
            ),
            Self::ExpectedLedgerHeadMismatch { expected, found } => write!(
                formatter,
                "durable enrollment allocation head mismatch: expected {expected}, found {found}"
            ),
            Self::Lifecycle(issues) => write!(
                formatter,
                "enrollment lifecycle validation failed with {} issue(s)",
                issues.len()
            ),
            Self::MissingSuccessorAllocation => write!(
                formatter,
                "allocation transition did not produce exactly one successor allocation"
            ),
            Self::ReadBackMismatch => {
                write!(formatter, "durable enrollment allocation read-back mismatch")
            }
            Self::Serialization => {
                write!(formatter, "durable enrollment allocation serialization failed")
            }
            Self::EntropyUnavailable => write!(formatter, "temporary-file entropy unavailable"),
            Self::LocalLockPoisoned => {
                write!(formatter, "durable enrollment allocation local lock poisoned")
            }
            Self::KernelLockUnavailable => {
                write!(formatter, "durable enrollment allocation kernel lock unavailable")
            }
            Self::Io(error) => write!(formatter, "durable enrollment allocation I/O failed: {error}"),
        }
    }
}

impl std::error::Error for DurableEnrollmentAllocationErrorV1 {}

impl From<std::io::Error> for DurableEnrollmentAllocationErrorV1 {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub struct DurableEnrollmentAllocationStoreV1 {
    root: PathBuf,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableEnrollmentAllocationStoreV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        &self,
        protocol: &FrozenPerceptualStudyProtocolV1,
        stimulus_pack: &FrozenPerceptualStimulusPackV1,
        render_binding: &FrozenC6fRenderSubjectBindingV1,
        cohort: &PerceptualCohortSlotsV1,
        token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
        identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
        schedule: &PerceptualParticipantScheduleBookV1,
        enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
        ledger: &FrozenPerceptualEnrollmentAllocationLedgerV1,
    ) -> Result<DurableEnrollmentAllocationStateV1, DurableEnrollmentAllocationErrorV1> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| DurableEnrollmentAllocationErrorV1::LocalLockPoisoned)?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelFileLock::exclusive(&lock_file)?;
        if self.state_path()?.exists() {
            return Err(DurableEnrollmentAllocationErrorV1::StateAlreadyExists);
        }
        let issues = validate_enrollment_allocation_ledger(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            &[],
            ledger,
        );
        if !issues.is_empty() {
            return Err(DurableEnrollmentAllocationErrorV1::Lifecycle(issues));
        }
        if !ledger.allocations.is_empty() {
            return Err(DurableEnrollmentAllocationErrorV1::StateIdentityMismatch);
        }
        let mut state = DurableEnrollmentAllocationStateV1 {
            state_version: DURABLE_ENROLLMENT_ALLOCATION_STATE_VERSION.into(),
            enrollment_policy_sha256: enrollment_policy.policy_sha256.clone(),
            token_generation_receipt_sha256: token_receipt.receipt_sha256.clone(),
            participant_schedule_sha256: canonical_json_sha256(schedule)
                .map_err(|_| DurableEnrollmentAllocationErrorV1::Serialization)?,
            eligibility_gates: Vec::new(),
            ledger: ledger.clone(),
            state_sha256: String::new(),
        };
        seal_state(&mut state)?;
        self.write_state_locked(&state)?;
        let read_back = self.read_state_locked()?;
        if read_back != state {
            return Err(DurableEnrollmentAllocationErrorV1::ReadBackMismatch);
        }
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inspect_current(
        &self,
        protocol: &FrozenPerceptualStudyProtocolV1,
        stimulus_pack: &FrozenPerceptualStimulusPackV1,
        render_binding: &FrozenC6fRenderSubjectBindingV1,
        cohort: &PerceptualCohortSlotsV1,
        token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
        identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
        schedule: &PerceptualParticipantScheduleBookV1,
        enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
        expected_ledger_sha256: &str,
    ) -> Result<DurableEnrollmentAllocationStateV1, DurableEnrollmentAllocationErrorV1> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| DurableEnrollmentAllocationErrorV1::LocalLockPoisoned)?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelFileLock::exclusive(&lock_file)?;
        let state = self.read_state_locked()?;
        self.validate_state_against_inputs(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            &state,
        )?;
        require_expected_head(expected_ledger_sha256, &state.ledger.ledger_sha256)?;
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn allocate_next(
        &self,
        protocol: &FrozenPerceptualStudyProtocolV1,
        stimulus_pack: &FrozenPerceptualStimulusPackV1,
        render_binding: &FrozenC6fRenderSubjectBindingV1,
        cohort: &PerceptualCohortSlotsV1,
        token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
        identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
        schedule: &PerceptualParticipantScheduleBookV1,
        enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
        expected_ledger_sha256: &str,
        gate: &FrozenPerceptualEligibilityGateReceiptV1,
    ) -> Result<UnwitnessedDurableEnrollmentAllocationV1, DurableEnrollmentAllocationErrorV1> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| DurableEnrollmentAllocationErrorV1::LocalLockPoisoned)?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelFileLock::exclusive(&lock_file)?;
        let mut state = self.read_state_locked()?;
        self.validate_state_against_inputs(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            &state,
        )?;
        require_expected_head(expected_ledger_sha256, &state.ledger.ledger_sha256)?;

        let previous_ledger_sha256 = state.ledger.ledger_sha256.clone();
        let previous_len = state.ledger.allocations.len();
        let projection = allocate_next_enrollment_slot(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            gate,
            &state.eligibility_gates,
            &mut state.ledger,
        )
        .map_err(DurableEnrollmentAllocationErrorV1::Lifecycle)?;
        if state.ledger.allocations.len() != previous_len.saturating_add(1) {
            return Err(DurableEnrollmentAllocationErrorV1::MissingSuccessorAllocation);
        }
        state.eligibility_gates.push(gate.clone());
        seal_state(&mut state)?;
        self.write_state_locked(&state)?;

        let read_back = self.read_state_locked()?;
        if read_back != state {
            return Err(DurableEnrollmentAllocationErrorV1::ReadBackMismatch);
        }
        self.validate_state_against_inputs(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            &read_back,
        )?;
        let allocation = read_back
            .ledger
            .allocations
            .last()
            .cloned()
            .ok_or(DurableEnrollmentAllocationErrorV1::MissingSuccessorAllocation)?;
        Ok(UnwitnessedDurableEnrollmentAllocationV1 {
            previous_ledger_sha256,
            current_ledger_sha256: read_back.ledger.ledger_sha256.clone(),
            current_state_sha256: read_back.state_sha256.clone(),
            allocation,
            participant_schedule_projection: projection,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_state_against_inputs(
        &self,
        protocol: &FrozenPerceptualStudyProtocolV1,
        stimulus_pack: &FrozenPerceptualStimulusPackV1,
        render_binding: &FrozenC6fRenderSubjectBindingV1,
        cohort: &PerceptualCohortSlotsV1,
        token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
        identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
        schedule: &PerceptualParticipantScheduleBookV1,
        enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
        state: &DurableEnrollmentAllocationStateV1,
    ) -> Result<(), DurableEnrollmentAllocationErrorV1> {
        if state.state_version != DURABLE_ENROLLMENT_ALLOCATION_STATE_VERSION
            || state.enrollment_policy_sha256 != enrollment_policy.policy_sha256
            || state.token_generation_receipt_sha256 != token_receipt.receipt_sha256
            || state.participant_schedule_sha256
                != canonical_json_sha256(schedule)
                    .map_err(|_| DurableEnrollmentAllocationErrorV1::Serialization)?
        {
            return Err(DurableEnrollmentAllocationErrorV1::StateIdentityMismatch);
        }
        if state.eligibility_gates.len() != state.ledger.allocations.len() {
            return Err(
                DurableEnrollmentAllocationErrorV1::GateAllocationCardinalityMismatch {
                    gates: state.eligibility_gates.len(),
                    allocations: state.ledger.allocations.len(),
                },
            );
        }
        for (index, (gate, allocation)) in state
            .eligibility_gates
            .iter()
            .zip(&state.ledger.allocations)
            .enumerate()
        {
            if gate.gate_sha256 != allocation.eligibility_gate_sha256 {
                return Err(DurableEnrollmentAllocationErrorV1::GateAllocationOrderMismatch {
                    index,
                });
            }
        }
        let issues = validate_enrollment_allocation_ledger(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            &state.eligibility_gates,
            &state.ledger,
        );
        if !issues.is_empty() {
            return Err(DurableEnrollmentAllocationErrorV1::Lifecycle(issues));
        }
        if state_commitment(state)? != state.state_sha256 {
            return Err(DurableEnrollmentAllocationErrorV1::StateDigestMismatch);
        }
        Ok(())
    }

    fn read_state_locked(
        &self,
    ) -> Result<DurableEnrollmentAllocationStateV1, DurableEnrollmentAllocationErrorV1> {
        let path = self.state_path()?;
        let file = match open_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(DurableEnrollmentAllocationErrorV1::StateMissing);
            }
            Err(error) => return Err(error.into()),
        };
        let metadata = file.metadata()?;
        if metadata.len() == 0 || metadata.len() > MAX_STATE_BYTES {
            return Err(DurableEnrollmentAllocationErrorV1::StateTooLarge);
        }
        let mut encoded = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut encoded)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_STATE_BYTES {
            return Err(DurableEnrollmentAllocationErrorV1::StateTooLarge);
        }
        let state: DurableEnrollmentAllocationStateV1 = serde_json::from_slice(&encoded)
            .map_err(|_| DurableEnrollmentAllocationErrorV1::StateMalformed)?;
        if state.state_version != DURABLE_ENROLLMENT_ALLOCATION_STATE_VERSION {
            return Err(DurableEnrollmentAllocationErrorV1::StateMalformed);
        }
        if state_commitment(&state)? != state.state_sha256 {
            return Err(DurableEnrollmentAllocationErrorV1::StateDigestMismatch);
        }
        Ok(state)
    }

    fn write_state_locked(
        &self,
        state: &DurableEnrollmentAllocationStateV1,
    ) -> Result<(), DurableEnrollmentAllocationErrorV1> {
        if state_commitment(state)? != state.state_sha256 {
            return Err(DurableEnrollmentAllocationErrorV1::StateDigestMismatch);
        }
        let encoded = canonical_json_bytes(state)
            .map_err(|_| DurableEnrollmentAllocationErrorV1::Serialization)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_STATE_BYTES {
            return Err(DurableEnrollmentAllocationErrorV1::StateTooLarge);
        }
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let mut nonce = [0u8; 16];
        OsRng
            .try_fill_bytes(&mut nonce)
            .map_err(|_| DurableEnrollmentAllocationErrorV1::EntropyUnavailable)?;
        let suffix = nonce
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let temp = operation_root.join(format!(
            ".mel003-enrollment-allocation-{}-{suffix}.tmp",
            std::process::id()
        ));
        let target = operation_root.join(STATE_FILE_NAME);
        let result = (|| {
            let mut file = open_regular_file(&temp, true, true)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            fs::rename(&temp, &target)?;
            root.sync_all()?;
            Ok::<(), DurableEnrollmentAllocationErrorV1>(())
        })();
        let _ = fs::remove_file(&temp);
        result
    }

    fn open_lock_file(&self) -> Result<File, DurableEnrollmentAllocationErrorV1> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_regular_file(&path, true, false).map_err(Into::into)
    }

    fn state_path(&self) -> Result<PathBuf, DurableEnrollmentAllocationErrorV1> {
        Ok(self.operation_root_path()?.join(STATE_FILE_NAME))
    }

    fn ensure_root(&self) -> Result<Arc<File>, DurableEnrollmentAllocationErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            return Err(DurableEnrollmentAllocationErrorV1::UnsupportedPlatform);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

            let mut pinned = self
                .pinned_root
                .lock()
                .map_err(|_| DurableEnrollmentAllocationErrorV1::LocalLockPoisoned)?;
            if let Some(root) = pinned.as_ref() {
                return Ok(Arc::clone(root));
            }
            fs::create_dir_all(&self.root)?;
            let metadata = fs::symlink_metadata(&self.root)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(DurableEnrollmentAllocationErrorV1::RootUnavailable);
            }
            fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
            let root = Arc::new(options.open(&self.root)?);
            if !root.metadata()?.is_dir() {
                return Err(DurableEnrollmentAllocationErrorV1::RootUnavailable);
            }
            *pinned = Some(Arc::clone(&root));
            Ok(root)
        }
    }

    fn operation_root_path(&self) -> Result<PathBuf, DurableEnrollmentAllocationErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            return Err(DurableEnrollmentAllocationErrorV1::UnsupportedPlatform);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;

            let root = self.ensure_root()?;
            let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
            if !path.is_dir() {
                return Err(DurableEnrollmentAllocationErrorV1::RootUnavailable);
            }
            Ok(path)
        }
    }
}

pub fn state_commitment(
    state: &DurableEnrollmentAllocationStateV1,
) -> Result<String, DurableEnrollmentAllocationErrorV1> {
    let mut unsigned = state.clone();
    unsigned.state_sha256.clear();
    canonical_json_sha256(&unsigned).map_err(|_| DurableEnrollmentAllocationErrorV1::Serialization)
}

pub fn seal_state(
    state: &mut DurableEnrollmentAllocationStateV1,
) -> Result<(), DurableEnrollmentAllocationErrorV1> {
    state.state_sha256 = state_commitment(state)?;
    Ok(())
}

fn require_expected_head(
    expected: &str,
    found: &str,
) -> Result<(), DurableEnrollmentAllocationErrorV1> {
    if expected == found {
        Ok(())
    } else {
        Err(DurableEnrollmentAllocationErrorV1::ExpectedLedgerHeadMismatch {
            expected: expected.into(),
            found: found.into(),
        })
    }
}

fn open_regular_file(path: &Path, create: bool, create_new: bool) -> std::io::Result<File> {
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
                "allocation state/lock must be a private regular file",
            ));
        }
        Ok(file)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, create, create_new);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "durable enrollment allocation requires Linux",
        ))
    }
}

struct KernelFileLock<'a> {
    file: &'a File,
}

impl<'a> KernelFileLock<'a> {
    fn exclusive(file: &'a File) -> Result<Self, DurableEnrollmentAllocationErrorV1> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;

            // SAFETY: `file` owns a valid descriptor for this guard's lifetime;
            // flock neither dereferences Rust pointers nor takes ownership.
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if result != 0 {
                return Err(DurableEnrollmentAllocationErrorV1::KernelLockUnavailable);
            }
            Ok(Self { file })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = file;
            Err(DurableEnrollmentAllocationErrorV1::UnsupportedPlatform)
        }
    }
}

impl Drop for KernelFileLock<'_> {
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
    fn state_commitment_detects_ledger_head_substitution() {
        let mut state = DurableEnrollmentAllocationStateV1 {
            state_version: DURABLE_ENROLLMENT_ALLOCATION_STATE_VERSION.into(),
            enrollment_policy_sha256: "a".repeat(64),
            token_generation_receipt_sha256: "b".repeat(64),
            participant_schedule_sha256: "c".repeat(64),
            eligibility_gates: Vec::new(),
            ledger: FrozenPerceptualEnrollmentAllocationLedgerV1 {
                ledger_version: "test".into(),
                enrollment_policy_sha256: "a".repeat(64),
                token_generation_receipt_sha256: "b".repeat(64),
                participant_schedule_sha256: "c".repeat(64),
                allocations: Vec::new(),
                final_allocation_head_sha256: "0".repeat(64),
                ledger_sha256: "d".repeat(64),
            },
            state_sha256: String::new(),
        };
        let first = state_commitment(&state).unwrap();
        state.ledger.ledger_sha256 = "e".repeat(64);
        assert_ne!(first, state_commitment(&state).unwrap());
    }

    #[test]
    fn local_success_type_is_explicitly_unwitnessed() {
        let value = std::any::type_name::<UnwitnessedDurableEnrollmentAllocationV1>();
        assert!(value.contains("UnwitnessedDurableEnrollmentAllocationV1"));
    }
}
