// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1RIR-B: crash-durable, idempotent recruitment-grant state.
//!
//! P1RIR-A owns signed recruitment-evidence semantics. This layer adds a local
//! runtime theorem: exact predecessor CAS for new semantic operations, exact
//! replay of already-persisted identical operations after restart, cross-process
//! serialization, atomic replacement, fsync, and read-back validation.
//!
//! The state contains only opaque P1RIR evidence. Raw recruitment identity stays
//! outside this store. This is local durability/integrity, not privileged-storage
//! anti-rollback evidence.

use crate::evidence_digest::{
    canonical_json_bytes, canonical_json_sha256,
    perceptual_recruitment_grant::{
        append_recruitment_grant_disposition, append_recruitment_grant_issuance,
        issue_recruitment_enrollment_grant_os_rng, validate_recruitment_grant_disposition_ledger,
        validate_recruitment_grant_issuance_register, validate_recruitment_policy,
        FrozenPerceptualRecruitmentPolicyV1, FrozenRecruitmentEnrollmentGrantV1,
        FrozenRecruitmentGrantDispositionLedgerV1, FrozenRecruitmentGrantIssuanceRegisterV1,
        PerceptualRecruitmentEvidenceIssueV1, RecruitmentGrantDispositionRecordV1,
        RecruitmentGrantDispositionV1, RecruitmentSigningKeyV1,
    },
};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

pub const DURABLE_RECRUITMENT_GRANT_STATE_VERSION: &str =
    "mel003-durable-recruitment-grant-state-v1";
const STATE_FILE_NAME: &str = "mel003-recruitment-grants.state.json";
const LOCK_FILE_NAME: &str = ".mel003-recruitment-grants.lock";
const MAX_STATE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableRecruitmentGrantStateV1 {
    pub state_version: String,
    pub recruitment_policy_sha256: String,
    pub issuance_register: FrozenRecruitmentGrantIssuanceRegisterV1,
    pub disposition_ledger: FrozenRecruitmentGrantDispositionLedgerV1,
    /// Integrity/correlation digest only; not an anti-rollback root against a
    /// privileged storage owner.
    pub state_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableRecruitmentGrantIssuanceV1 {
    pub replayed_existing_operation: bool,
    pub previous_state_sha256: String,
    pub current_state_sha256: String,
    pub current_issuance_register_sha256: String,
    pub grant: FrozenRecruitmentEnrollmentGrantV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableRecruitmentGrantTerminalizationV1 {
    pub replayed_existing_operation: bool,
    pub previous_state_sha256: String,
    pub current_state_sha256: String,
    pub current_disposition_ledger_sha256: String,
    pub record: RecruitmentGrantDispositionRecordV1,
}

#[derive(Debug)]
pub enum DurableRecruitmentGrantErrorV1 {
    UnsupportedPlatform,
    RootUnavailable,
    StateAlreadyExists,
    StateMissing,
    StateTooLarge,
    StateMalformed,
    StateIdentityMismatch,
    StateDigestMismatch,
    ExpectedStateHeadMismatch { expected: String, found: String },
    Evidence(Vec<PerceptualRecruitmentEvidenceIssueV1>),
    DuplicateEligibilityAttempt { eligibility_attempt_sha256: String },
    DuplicateIssuanceChronology { chronology_event_sha256: String },
    DuplicateDispositionChronology { chronology_event_sha256: String },
    IssuanceRequestConflict { eligibility_attempt_sha256: String },
    UnknownGrant { grant_sha256: String },
    TerminalizationConflict { grant_sha256: String },
    MissingSuccessor,
    ReadBackMismatch,
    Serialization,
    EntropyUnavailable,
    LocalLockPoisoned,
    KernelLockUnavailable,
    Io(std::io::Error),
}

impl std::fmt::Display for DurableRecruitmentGrantErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(f, "durable recruitment store requires Linux"),
            Self::RootUnavailable => write!(f, "durable recruitment root unavailable"),
            Self::StateAlreadyExists => write!(f, "durable recruitment state already exists"),
            Self::StateMissing => write!(f, "durable recruitment state missing"),
            Self::StateTooLarge => write!(f, "durable recruitment state exceeds bound"),
            Self::StateMalformed => write!(f, "durable recruitment state malformed"),
            Self::StateIdentityMismatch => write!(f, "durable recruitment state identity mismatch"),
            Self::StateDigestMismatch => write!(f, "durable recruitment state digest mismatch"),
            Self::ExpectedStateHeadMismatch { expected, found } => write!(
                f,
                "durable recruitment state head mismatch: expected {expected}, found {found}"
            ),
            Self::Evidence(issues) => write!(
                f,
                "recruitment evidence validation failed with {} issue(s)",
                issues.len()
            ),
            Self::DuplicateEligibilityAttempt { eligibility_attempt_sha256 } => write!(
                f,
                "duplicate eligibility attempt in durable issuance state: {eligibility_attempt_sha256}"
            ),
            Self::DuplicateIssuanceChronology { chronology_event_sha256 } => write!(
                f,
                "duplicate issuance chronology event: {chronology_event_sha256}"
            ),
            Self::DuplicateDispositionChronology { chronology_event_sha256 } => write!(
                f,
                "duplicate terminal chronology event: {chronology_event_sha256}"
            ),
            Self::IssuanceRequestConflict { eligibility_attempt_sha256 } => write!(
                f,
                "eligibility attempt already issued under a different request: {eligibility_attempt_sha256}"
            ),
            Self::UnknownGrant { grant_sha256 } => {
                write!(f, "grant is absent from durable issuance state: {grant_sha256}")
            }
            Self::TerminalizationConflict { grant_sha256 } => write!(
                f,
                "grant already has a different terminal request: {grant_sha256}"
            ),
            Self::MissingSuccessor => write!(f, "transition did not produce exactly one successor"),
            Self::ReadBackMismatch => write!(f, "durable recruitment read-back mismatch"),
            Self::Serialization => write!(f, "durable recruitment serialization failed"),
            Self::EntropyUnavailable => write!(f, "temporary-file entropy unavailable"),
            Self::LocalLockPoisoned => write!(f, "durable recruitment local lock poisoned"),
            Self::KernelLockUnavailable => write!(f, "durable recruitment kernel lock unavailable"),
            Self::Io(error) => write!(f, "durable recruitment I/O failed: {error}"),
        }
    }
}

impl std::error::Error for DurableRecruitmentGrantErrorV1 {}

impl From<std::io::Error> for DurableRecruitmentGrantErrorV1 {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub struct DurableRecruitmentGrantStoreV1 {
    root: PathBuf,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableRecruitmentGrantStoreV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        }
    }

    pub fn initialize(
        &self,
        policy: &FrozenPerceptualRecruitmentPolicyV1,
        issuance_register: &FrozenRecruitmentGrantIssuanceRegisterV1,
        disposition_ledger: &FrozenRecruitmentGrantDispositionLedgerV1,
    ) -> Result<DurableRecruitmentGrantStateV1, DurableRecruitmentGrantErrorV1> {
        let _local = self.lock_local()?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelRecruitmentLock::exclusive(&lock_file)?;
        if self.state_path()?.exists() {
            return Err(DurableRecruitmentGrantErrorV1::StateAlreadyExists);
        }
        let policy_issues = validate_recruitment_policy(policy);
        if !policy_issues.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::Evidence(policy_issues));
        }
        let register_issues = validate_recruitment_grant_issuance_register(policy, issuance_register);
        if !register_issues.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::Evidence(register_issues));
        }
        let ledger_issues = validate_recruitment_grant_disposition_ledger(
            policy,
            issuance_register,
            disposition_ledger,
        );
        if !ledger_issues.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::Evidence(ledger_issues));
        }
        if !issuance_register.entries.is_empty() || !disposition_ledger.entries.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::StateIdentityMismatch);
        }
        let mut state = DurableRecruitmentGrantStateV1 {
            state_version: DURABLE_RECRUITMENT_GRANT_STATE_VERSION.into(),
            recruitment_policy_sha256: policy.policy_sha256.clone(),
            issuance_register: issuance_register.clone(),
            disposition_ledger: disposition_ledger.clone(),
            state_sha256: String::new(),
        };
        seal_recruitment_state(&mut state)?;
        self.write_state_locked(&state)?;
        let read_back = self.read_state_locked()?;
        if read_back != state {
            return Err(DurableRecruitmentGrantErrorV1::ReadBackMismatch);
        }
        self.validate_state(policy, &read_back)?;
        Ok(read_back)
    }

    pub fn inspect_current(
        &self,
        policy: &FrozenPerceptualRecruitmentPolicyV1,
        expected_state_sha256: &str,
    ) -> Result<DurableRecruitmentGrantStateV1, DurableRecruitmentGrantErrorV1> {
        let _local = self.lock_local()?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelRecruitmentLock::exclusive(&lock_file)?;
        let state = self.read_state_locked()?;
        self.validate_state(policy, &state)?;
        require_expected_state(expected_state_sha256, &state.state_sha256)?;
        Ok(state)
    }

    /// Idempotency key: `eligibility_attempt_sha256`.
    ///
    /// Exact replay is allowed before expected-state CAS. Any changed request for
    /// that attempt conflicts. Any new attempt must pass exact predecessor CAS.
    pub fn issue_grant(
        &self,
        policy: &FrozenPerceptualRecruitmentPolicyV1,
        signing_key: &RecruitmentSigningKeyV1,
        expected_state_sha256: &str,
        eligibility_attempt_sha256: &str,
        issued_chronology_event_sha256: &str,
    ) -> Result<DurableRecruitmentGrantIssuanceV1, DurableRecruitmentGrantErrorV1> {
        let _local = self.lock_local()?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelRecruitmentLock::exclusive(&lock_file)?;
        let mut state = self.read_state_locked()?;
        self.validate_state(policy, &state)?;
        if signing_key.verifier_identity() != policy.recruitment_authority {
            return Err(DurableRecruitmentGrantErrorV1::Evidence(vec![
                PerceptualRecruitmentEvidenceIssueV1::GrantAuthorityIdentityMismatch,
            ]));
        }

        let matching: Vec<_> = state
            .issuance_register
            .entries
            .iter()
            .filter(|entry| entry.grant.eligibility_attempt_sha256 == eligibility_attempt_sha256)
            .collect();
        if let [existing] = matching.as_slice() {
            if existing.grant.issued_chronology_event_sha256 == issued_chronology_event_sha256 {
                return Ok(DurableRecruitmentGrantIssuanceV1 {
                    replayed_existing_operation: true,
                    previous_state_sha256: state.state_sha256.clone(),
                    current_state_sha256: state.state_sha256.clone(),
                    current_issuance_register_sha256: state.issuance_register.register_sha256.clone(),
                    grant: existing.grant.clone(),
                });
            }
        }
        if !matching.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::IssuanceRequestConflict {
                eligibility_attempt_sha256: eligibility_attempt_sha256.into(),
            });
        }
        if state.issuance_register.entries.iter().any(|entry| {
            entry.grant.issued_chronology_event_sha256 == issued_chronology_event_sha256
        }) {
            return Err(DurableRecruitmentGrantErrorV1::DuplicateIssuanceChronology {
                chronology_event_sha256: issued_chronology_event_sha256.into(),
            });
        }

        require_expected_state(expected_state_sha256, &state.state_sha256)?;
        let previous_state_sha256 = state.state_sha256.clone();
        let previous_len = state.issuance_register.entries.len();
        let grant = issue_recruitment_enrollment_grant_os_rng(
            policy,
            signing_key,
            eligibility_attempt_sha256,
            issued_chronology_event_sha256,
        )
        .map_err(DurableRecruitmentGrantErrorV1::Evidence)?;
        append_recruitment_grant_issuance(policy, &mut state.issuance_register, grant.clone())
            .map_err(DurableRecruitmentGrantErrorV1::Evidence)?;
        if state.issuance_register.entries.len() != previous_len.saturating_add(1) {
            return Err(DurableRecruitmentGrantErrorV1::MissingSuccessor);
        }
        seal_recruitment_state(&mut state)?;
        self.write_state_locked(&state)?;
        let read_back = self.read_state_locked()?;
        if read_back != state {
            return Err(DurableRecruitmentGrantErrorV1::ReadBackMismatch);
        }
        self.validate_state(policy, &read_back)?;
        let persisted = read_back
            .issuance_register
            .entries
            .last()
            .ok_or(DurableRecruitmentGrantErrorV1::MissingSuccessor)?
            .grant
            .clone();
        Ok(DurableRecruitmentGrantIssuanceV1 {
            replayed_existing_operation: false,
            previous_state_sha256,
            current_state_sha256: read_back.state_sha256.clone(),
            current_issuance_register_sha256: read_back.issuance_register.register_sha256.clone(),
            grant: persisted,
        })
    }

    /// Exact replay is allowed before expected-state CAS. A changed request for
    /// an already-terminal grant conflicts. A first terminalization requires CAS.
    #[allow(clippy::too_many_arguments)]
    pub fn terminalize_grant(
        &self,
        policy: &FrozenPerceptualRecruitmentPolicyV1,
        expected_state_sha256: &str,
        grant_sha256: &str,
        disposition: RecruitmentGrantDispositionV1,
        eligibility_gate_sha256: Option<String>,
        disposition_chronology_event_sha256: &str,
    ) -> Result<DurableRecruitmentGrantTerminalizationV1, DurableRecruitmentGrantErrorV1> {
        let _local = self.lock_local()?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelRecruitmentLock::exclusive(&lock_file)?;
        let mut state = self.read_state_locked()?;
        self.validate_state(policy, &state)?;

        let grant = state
            .issuance_register
            .entries
            .iter()
            .find(|entry| entry.grant.grant_sha256 == grant_sha256)
            .map(|entry| entry.grant.clone())
            .ok_or_else(|| DurableRecruitmentGrantErrorV1::UnknownGrant {
                grant_sha256: grant_sha256.into(),
            })?;

        let existing: Vec<_> = state
            .disposition_ledger
            .entries
            .iter()
            .filter(|entry| entry.grant_sha256 == grant_sha256)
            .collect();
        if let [record] = existing.as_slice() {
            if record.disposition == disposition
                && record.eligibility_gate_sha256 == eligibility_gate_sha256
                && record.disposition_chronology_event_sha256 == disposition_chronology_event_sha256
            {
                return Ok(DurableRecruitmentGrantTerminalizationV1 {
                    replayed_existing_operation: true,
                    previous_state_sha256: state.state_sha256.clone(),
                    current_state_sha256: state.state_sha256.clone(),
                    current_disposition_ledger_sha256: state.disposition_ledger.ledger_sha256.clone(),
                    record: (*record).clone(),
                });
            }
        }
        if !existing.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::TerminalizationConflict {
                grant_sha256: grant_sha256.into(),
            });
        }
        if state.disposition_ledger.entries.iter().any(|entry| {
            entry.disposition_chronology_event_sha256 == disposition_chronology_event_sha256
        }) {
            return Err(DurableRecruitmentGrantErrorV1::DuplicateDispositionChronology {
                chronology_event_sha256: disposition_chronology_event_sha256.into(),
            });
        }

        require_expected_state(expected_state_sha256, &state.state_sha256)?;
        let previous_state_sha256 = state.state_sha256.clone();
        let previous_len = state.disposition_ledger.entries.len();
        append_recruitment_grant_disposition(
            policy,
            &state.issuance_register,
            &mut state.disposition_ledger,
            &grant,
            disposition,
            eligibility_gate_sha256,
            disposition_chronology_event_sha256,
        )
        .map_err(DurableRecruitmentGrantErrorV1::Evidence)?;
        if state.disposition_ledger.entries.len() != previous_len.saturating_add(1) {
            return Err(DurableRecruitmentGrantErrorV1::MissingSuccessor);
        }
        seal_recruitment_state(&mut state)?;
        self.write_state_locked(&state)?;
        let read_back = self.read_state_locked()?;
        if read_back != state {
            return Err(DurableRecruitmentGrantErrorV1::ReadBackMismatch);
        }
        self.validate_state(policy, &read_back)?;
        let record = read_back
            .disposition_ledger
            .entries
            .last()
            .cloned()
            .ok_or(DurableRecruitmentGrantErrorV1::MissingSuccessor)?;
        Ok(DurableRecruitmentGrantTerminalizationV1 {
            replayed_existing_operation: false,
            previous_state_sha256,
            current_state_sha256: read_back.state_sha256.clone(),
            current_disposition_ledger_sha256: read_back.disposition_ledger.ledger_sha256.clone(),
            record,
        })
    }

    fn validate_state(
        &self,
        policy: &FrozenPerceptualRecruitmentPolicyV1,
        state: &DurableRecruitmentGrantStateV1,
    ) -> Result<(), DurableRecruitmentGrantErrorV1> {
        if state.state_version != DURABLE_RECRUITMENT_GRANT_STATE_VERSION
            || state.recruitment_policy_sha256 != policy.policy_sha256
        {
            return Err(DurableRecruitmentGrantErrorV1::StateIdentityMismatch);
        }
        let register_issues =
            validate_recruitment_grant_issuance_register(policy, &state.issuance_register);
        if !register_issues.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::Evidence(register_issues));
        }
        let ledger_issues = validate_recruitment_grant_disposition_ledger(
            policy,
            &state.issuance_register,
            &state.disposition_ledger,
        );
        if !ledger_issues.is_empty() {
            return Err(DurableRecruitmentGrantErrorV1::Evidence(ledger_issues));
        }
        let mut attempts = BTreeSet::new();
        let mut issuance_chronology = BTreeSet::new();
        for entry in &state.issuance_register.entries {
            if !attempts.insert(entry.grant.eligibility_attempt_sha256.as_str()) {
                return Err(DurableRecruitmentGrantErrorV1::DuplicateEligibilityAttempt {
                    eligibility_attempt_sha256: entry.grant.eligibility_attempt_sha256.clone(),
                });
            }
            if !issuance_chronology.insert(entry.grant.issued_chronology_event_sha256.as_str()) {
                return Err(DurableRecruitmentGrantErrorV1::DuplicateIssuanceChronology {
                    chronology_event_sha256: entry.grant.issued_chronology_event_sha256.clone(),
                });
            }
        }
        let mut disposition_chronology = BTreeSet::new();
        for entry in &state.disposition_ledger.entries {
            if !disposition_chronology.insert(entry.disposition_chronology_event_sha256.as_str()) {
                return Err(DurableRecruitmentGrantErrorV1::DuplicateDispositionChronology {
                    chronology_event_sha256: entry.disposition_chronology_event_sha256.clone(),
                });
            }
        }
        if recruitment_state_commitment(state)? != state.state_sha256 {
            return Err(DurableRecruitmentGrantErrorV1::StateDigestMismatch);
        }
        Ok(())
    }

    fn read_state_locked(
        &self,
    ) -> Result<DurableRecruitmentGrantStateV1, DurableRecruitmentGrantErrorV1> {
        let path = self.state_path()?;
        let file = match open_private_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(DurableRecruitmentGrantErrorV1::StateMissing);
            }
            Err(error) => return Err(error.into()),
        };
        let metadata = file.metadata()?;
        if metadata.len() == 0 || metadata.len() > MAX_STATE_BYTES {
            return Err(DurableRecruitmentGrantErrorV1::StateTooLarge);
        }
        let mut encoded = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut encoded)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_STATE_BYTES {
            return Err(DurableRecruitmentGrantErrorV1::StateTooLarge);
        }
        let state: DurableRecruitmentGrantStateV1 = serde_json::from_slice(&encoded)
            .map_err(|_| DurableRecruitmentGrantErrorV1::StateMalformed)?;
        if state.state_version != DURABLE_RECRUITMENT_GRANT_STATE_VERSION {
            return Err(DurableRecruitmentGrantErrorV1::StateMalformed);
        }
        if recruitment_state_commitment(&state)? != state.state_sha256 {
            return Err(DurableRecruitmentGrantErrorV1::StateDigestMismatch);
        }
        Ok(state)
    }

    fn write_state_locked(
        &self,
        state: &DurableRecruitmentGrantStateV1,
    ) -> Result<(), DurableRecruitmentGrantErrorV1> {
        if recruitment_state_commitment(state)? != state.state_sha256 {
            return Err(DurableRecruitmentGrantErrorV1::StateDigestMismatch);
        }
        let encoded = canonical_json_bytes(state)
            .map_err(|_| DurableRecruitmentGrantErrorV1::Serialization)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_STATE_BYTES {
            return Err(DurableRecruitmentGrantErrorV1::StateTooLarge);
        }
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let mut nonce = [0u8; 16];
        OsRng
            .try_fill_bytes(&mut nonce)
            .map_err(|_| DurableRecruitmentGrantErrorV1::EntropyUnavailable)?;
        let suffix = nonce
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let temp = operation_root.join(format!(
            ".mel003-recruitment-grants-{}-{suffix}.tmp",
            std::process::id()
        ));
        let target = operation_root.join(STATE_FILE_NAME);
        let result = (|| {
            let mut file = open_private_regular_file(&temp, true, true)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            fs::rename(&temp, &target)?;
            root.sync_all()?;
            Ok::<(), DurableRecruitmentGrantErrorV1>(())
        })();
        let _ = fs::remove_file(&temp);
        result
    }

    fn lock_local(&self) -> Result<MutexGuard<'_, ()>, DurableRecruitmentGrantErrorV1> {
        self.local_lock
            .lock()
            .map_err(|_| DurableRecruitmentGrantErrorV1::LocalLockPoisoned)
    }

    fn open_lock_file(&self) -> Result<File, DurableRecruitmentGrantErrorV1> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_private_regular_file(&path, true, false).map_err(Into::into)
    }

    fn state_path(&self) -> Result<PathBuf, DurableRecruitmentGrantErrorV1> {
        Ok(self.operation_root_path()?.join(STATE_FILE_NAME))
    }

    fn ensure_root(&self) -> Result<Arc<File>, DurableRecruitmentGrantErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            return Err(DurableRecruitmentGrantErrorV1::UnsupportedPlatform);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            let mut pinned = self
                .pinned_root
                .lock()
                .map_err(|_| DurableRecruitmentGrantErrorV1::LocalLockPoisoned)?;
            if let Some(root) = pinned.as_ref() {
                return Ok(Arc::clone(root));
            }
            fs::create_dir_all(&self.root)?;
            let metadata = fs::symlink_metadata(&self.root)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(DurableRecruitmentGrantErrorV1::RootUnavailable);
            }
            fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
            let root = Arc::new(options.open(&self.root)?);
            if !root.metadata()?.is_dir() {
                return Err(DurableRecruitmentGrantErrorV1::RootUnavailable);
            }
            *pinned = Some(Arc::clone(&root));
            Ok(root)
        }
    }

    fn operation_root_path(&self) -> Result<PathBuf, DurableRecruitmentGrantErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            Err(DurableRecruitmentGrantErrorV1::UnsupportedPlatform)
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let root = self.ensure_root()?;
            let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
            if !path.is_dir() {
                return Err(DurableRecruitmentGrantErrorV1::RootUnavailable);
            }
            Ok(path)
        }
    }
}

pub fn recruitment_state_commitment(
    state: &DurableRecruitmentGrantStateV1,
) -> Result<String, DurableRecruitmentGrantErrorV1> {
    let mut unsigned = state.clone();
    unsigned.state_sha256.clear();
    canonical_json_sha256(&unsigned).map_err(|_| DurableRecruitmentGrantErrorV1::Serialization)
}

pub fn seal_recruitment_state(
    state: &mut DurableRecruitmentGrantStateV1,
) -> Result<(), DurableRecruitmentGrantErrorV1> {
    state.state_sha256 = recruitment_state_commitment(state)?;
    Ok(())
}

fn require_expected_state(
    expected: &str,
    found: &str,
) -> Result<(), DurableRecruitmentGrantErrorV1> {
    if expected == found {
        Ok(())
    } else {
        Err(DurableRecruitmentGrantErrorV1::ExpectedStateHeadMismatch {
            expected: expected.into(),
            found: found.into(),
        })
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
                "recruitment state/lock must be a private regular file",
            ));
        }
        Ok(file)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, create, create_new);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "durable recruitment store requires Linux",
        ))
    }
}

struct KernelRecruitmentLock<'a> {
    file: &'a File,
}

impl<'a> KernelRecruitmentLock<'a> {
    fn exclusive(file: &'a File) -> Result<Self, DurableRecruitmentGrantErrorV1> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: the guard borrows a live descriptor for its full lifetime.
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if result != 0 {
                return Err(DurableRecruitmentGrantErrorV1::KernelLockUnavailable);
            }
            Ok(Self { file })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = file;
            Err(DurableRecruitmentGrantErrorV1::UnsupportedPlatform)
        }
    }
}

impl Drop for KernelRecruitmentLock<'_> {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: unlock the same descriptor retained by this guard.
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::perceptual_recruitment_grant::{
        new_recruitment_grant_disposition_ledger, new_recruitment_grant_issuance_register,
        seal_recruitment_policy, DuplicateEnrollmentClaimCeilingV1,
        RestrictedRecruitmentLinkageMechanismV1, PERCEPTUAL_RECRUITMENT_POLICY_VERSION,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_ROOT: AtomicU64 = AtomicU64::new(1);

    fn digest(marker: u8) -> String {
        format!("{marker:02x}").repeat(32)
    }

    fn root() -> PathBuf {
        let serial = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or_default();
        std::env::temp_dir().join(format!(
            "symthaea-mel003-recruitment-store-{}-{nanos}-{serial}",
            std::process::id()
        ))
    }

    fn policy_and_key() -> (FrozenPerceptualRecruitmentPolicyV1, RecruitmentSigningKeyV1) {
        let key =
            RecruitmentSigningKeyV1::from_seed("recruitment-store-test", 1, [0x55; 32]).unwrap();
        let mut policy = FrozenPerceptualRecruitmentPolicyV1 {
            policy_version: PERCEPTUAL_RECRUITMENT_POLICY_VERSION.into(),
            protocol_sha256: digest(1),
            recruitment_material_sha256: digest(2),
            participant_information_sha256: digest(3),
            privacy_notice_sha256: digest(4),
            linkage_retention_deletion_policy_sha256: digest(5),
            linkage_mechanism: RestrictedRecruitmentLinkageMechanismV1::ExistingRecruitmentAccount,
            duplicate_enrollment_claim_ceiling:
                DuplicateEnrollmentClaimCeilingV1::NoDuplicateAcceptedUnderFrozenMechanism,
            recruitment_authority: key.verifier_identity(),
            direct_identity_export_to_study_evidence_prohibited: true,
            private_arm_mapping_access_prohibited: true,
            scored_response_access_prohibited: true,
            correctness_or_significance_access_prohibited: true,
            recruitment_service_schedule_choice_prohibited: true,
            enrollment_grant_reuse_prohibited: true,
            post_enrollment_grant_reissue_prohibited: true,
            policy_sha256: String::new(),
        };
        seal_recruitment_policy(&mut policy).unwrap();
        (policy, key)
    }

    fn initialized() -> (
        PathBuf,
        FrozenPerceptualRecruitmentPolicyV1,
        RecruitmentSigningKeyV1,
        DurableRecruitmentGrantStateV1,
    ) {
        let root = root();
        let store = DurableRecruitmentGrantStoreV1::new(&root);
        let (policy, key) = policy_and_key();
        let register = new_recruitment_grant_issuance_register(&policy).unwrap();
        let ledger = new_recruitment_grant_disposition_ledger(&policy, &register).unwrap();
        let state = store.initialize(&policy, &register, &ledger).unwrap();
        (root, policy, key, state)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn issuance_replays_exactly_after_store_restart() {
        let (root, policy, key, initial) = initialized();
        let store = DurableRecruitmentGrantStoreV1::new(&root);
        let first = store
            .issue_grant(&policy, &key, &initial.state_sha256, &digest(6), &digest(7))
            .unwrap();
        assert!(!first.replayed_existing_operation);
        drop(store);
        let restarted = DurableRecruitmentGrantStoreV1::new(&root);
        let replay = restarted
            .issue_grant(&policy, &key, &initial.state_sha256, &digest(6), &digest(7))
            .unwrap();
        assert!(replay.replayed_existing_operation);
        assert_eq!(replay.grant, first.grant);
        assert_eq!(replay.current_state_sha256, first.current_state_sha256);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn changed_or_reused_issuance_request_fails_before_write() {
        let (root, policy, key, initial) = initialized();
        let store = DurableRecruitmentGrantStoreV1::new(&root);
        let first = store
            .issue_grant(&policy, &key, &initial.state_sha256, &digest(6), &digest(7))
            .unwrap();
        let conflict = store
            .issue_grant(&policy, &key, &first.current_state_sha256, &digest(6), &digest(8))
            .unwrap_err();
        assert!(matches!(
            conflict,
            DurableRecruitmentGrantErrorV1::IssuanceRequestConflict { .. }
        ));
        let chronology_reuse = store
            .issue_grant(&policy, &key, &first.current_state_sha256, &digest(8), &digest(7))
            .unwrap_err();
        assert!(matches!(
            chronology_reuse,
            DurableRecruitmentGrantErrorV1::DuplicateIssuanceChronology { .. }
        ));
        let current = store.inspect_current(&policy, &first.current_state_sha256).unwrap();
        assert_eq!(current.issuance_register.entries.len(), 1);
        let stale = store
            .issue_grant(&policy, &key, &initial.state_sha256, &digest(9), &digest(10))
            .unwrap_err();
        assert!(matches!(
            stale,
            DurableRecruitmentGrantErrorV1::ExpectedStateHeadMismatch { .. }
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn terminalization_replays_exactly_but_changed_request_fails() {
        let (root, policy, key, initial) = initialized();
        let store = DurableRecruitmentGrantStoreV1::new(&root);
        let issuance = store
            .issue_grant(&policy, &key, &initial.state_sha256, &digest(6), &digest(7))
            .unwrap();
        let terminal = store
            .terminalize_grant(
                &policy,
                &issuance.current_state_sha256,
                &issuance.grant.grant_sha256,
                RecruitmentGrantDispositionV1::ConsumedForEnrollment,
                Some(digest(8)),
                &digest(9),
            )
            .unwrap();
        drop(store);
        let restarted = DurableRecruitmentGrantStoreV1::new(&root);
        let replay = restarted
            .terminalize_grant(
                &policy,
                &issuance.current_state_sha256,
                &issuance.grant.grant_sha256,
                RecruitmentGrantDispositionV1::ConsumedForEnrollment,
                Some(digest(8)),
                &digest(9),
            )
            .unwrap();
        assert!(replay.replayed_existing_operation);
        assert_eq!(replay.record, terminal.record);
        assert_eq!(replay.current_state_sha256, terminal.current_state_sha256);
        let changed = restarted
            .terminalize_grant(
                &policy,
                &terminal.current_state_sha256,
                &issuance.grant.grant_sha256,
                RecruitmentGrantDispositionV1::WithdrawnBeforeEnrollment,
                None,
                &digest(10),
            )
            .unwrap_err();
        assert!(matches!(
            changed,
            DurableRecruitmentGrantErrorV1::TerminalizationConflict { .. }
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn state_commitment_detects_register_substitution() {
        let (policy, _) = policy_and_key();
        let register = new_recruitment_grant_issuance_register(&policy).unwrap();
        let ledger = new_recruitment_grant_disposition_ledger(&policy, &register).unwrap();
        let mut state = DurableRecruitmentGrantStateV1 {
            state_version: DURABLE_RECRUITMENT_GRANT_STATE_VERSION.into(),
            recruitment_policy_sha256: policy.policy_sha256,
            issuance_register: register,
            disposition_ledger: ledger,
            state_sha256: String::new(),
        };
        let first = recruitment_state_commitment(&state).unwrap();
        state.issuance_register.register_sha256 = digest(15);
        assert_ne!(first, recruitment_state_commitment(&state).unwrap());
    }
}
