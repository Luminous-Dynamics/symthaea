// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate-independent atomic activation transaction contract.
//!
//! EKM-071 can prove only that an activation transaction design is eligible for
//! review. This module defines the ordering and state-component invariants such a
//! future transaction must satisfy. It contains no executor, lock, compare-and-
//! swap primitive, rollback implementation, live-state mutation, or checkpoint
//! commit authority.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicActivationTransactionContractVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomicActivationTransactionContractDigestV1([u8; 32]);

impl AtomicActivationTransactionContractDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

/// State that must move as one coherent activation bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicActivationStateComponentV1 {
    EpistemicLedger,
    EpistemicSupportStore,
    OperationalRevisionHistory,
    RevisionSchemaHistory,
    MutationAuthorizationConsumptionState,
}

impl AtomicActivationStateComponentV1 {
    fn tag(self) -> u8 {
        match self {
            Self::EpistemicLedger => 1,
            Self::EpistemicSupportStore => 2,
            Self::OperationalRevisionHistory => 3,
            Self::RevisionSchemaHistory => 4,
            Self::MutationAuthorizationConsumptionState => 5,
        }
    }
}

/// Required ordering for a future activation transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicActivationPhaseV1 {
    AcquireExclusiveLiveEpochGuard,
    ReverifyActivationReviewUnderGuard,
    CompareExpectedLiveEpoch,
    CaptureRollbackBundle,
    StageCompleteCandidateBundle,
    AtomicLiveStateSwap,
    VerifyInstalledStateUnderGuard,
    CommitTrustedCheckpoints,
    ReleaseLiveEpochGuard,
}

impl AtomicActivationPhaseV1 {
    fn tag(self) -> u8 {
        match self {
            Self::AcquireExclusiveLiveEpochGuard => 1,
            Self::ReverifyActivationReviewUnderGuard => 2,
            Self::CompareExpectedLiveEpoch => 3,
            Self::CaptureRollbackBundle => 4,
            Self::StageCompleteCandidateBundle => 5,
            Self::AtomicLiveStateSwap => 6,
            Self::VerifyInstalledStateUnderGuard => 7,
            Self::CommitTrustedCheckpoints => 8,
            Self::ReleaseLiveEpochGuard => 9,
        }
    }
}

/// Canonical transaction contract. All authority flags describe what this module
/// itself implements, not what a future implementation may eventually provide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicActivationTransactionContractV1 {
    version: AtomicActivationTransactionContractVersion,
    required_components: Vec<AtomicActivationStateComponentV1>,
    required_phases: Vec<AtomicActivationPhaseV1>,
    exclusive_live_epoch_guard_required: bool,
    under_guard_reverification_required: bool,
    compare_and_swap_required: bool,
    rollback_bundle_required_before_swap: bool,
    post_swap_equivalence_required_before_checkpoint_commit: bool,
    trusted_checkpoint_commit_must_follow_installed_state_verification: bool,
    partial_bundle_swap_allowed: bool,
    partial_checkpoint_commit_allowed: bool,
    executor_implemented: bool,
    live_state_lock_implemented: bool,
    compare_and_swap_implemented: bool,
    rollback_implemented: bool,
    live_state_swap_authorized: bool,
    activation_authorized: bool,
    trusted_checkpoint_commit_authorized: bool,
    contract_digest: AtomicActivationTransactionContractDigestV1,
}

impl AtomicActivationTransactionContractV1 {
    /// Return the only canonical V1 contract.
    pub fn canonical() -> Self {
        let mut out = Self {
            version: AtomicActivationTransactionContractVersion::V1,
            required_components: vec![
                AtomicActivationStateComponentV1::EpistemicLedger,
                AtomicActivationStateComponentV1::EpistemicSupportStore,
                AtomicActivationStateComponentV1::OperationalRevisionHistory,
                AtomicActivationStateComponentV1::RevisionSchemaHistory,
                AtomicActivationStateComponentV1::MutationAuthorizationConsumptionState,
            ],
            required_phases: vec![
                AtomicActivationPhaseV1::AcquireExclusiveLiveEpochGuard,
                AtomicActivationPhaseV1::ReverifyActivationReviewUnderGuard,
                AtomicActivationPhaseV1::CompareExpectedLiveEpoch,
                AtomicActivationPhaseV1::CaptureRollbackBundle,
                AtomicActivationPhaseV1::StageCompleteCandidateBundle,
                AtomicActivationPhaseV1::AtomicLiveStateSwap,
                AtomicActivationPhaseV1::VerifyInstalledStateUnderGuard,
                AtomicActivationPhaseV1::CommitTrustedCheckpoints,
                AtomicActivationPhaseV1::ReleaseLiveEpochGuard,
            ],
            exclusive_live_epoch_guard_required: true,
            under_guard_reverification_required: true,
            compare_and_swap_required: true,
            rollback_bundle_required_before_swap: true,
            post_swap_equivalence_required_before_checkpoint_commit: true,
            trusted_checkpoint_commit_must_follow_installed_state_verification: true,
            partial_bundle_swap_allowed: false,
            partial_checkpoint_commit_allowed: false,
            executor_implemented: false,
            live_state_lock_implemented: false,
            compare_and_swap_implemented: false,
            rollback_implemented: false,
            live_state_swap_authorized: false,
            activation_authorized: false,
            trusted_checkpoint_commit_authorized: false,
            contract_digest: AtomicActivationTransactionContractDigestV1([0; 32]),
        };
        out.contract_digest = digest_contract(&out);
        out
    }

    pub fn version(&self) -> AtomicActivationTransactionContractVersion {
        self.version
    }

    pub fn required_components(&self) -> &[AtomicActivationStateComponentV1] {
        &self.required_components
    }

    pub fn required_phases(&self) -> &[AtomicActivationPhaseV1] {
        &self.required_phases
    }

    pub fn exclusive_live_epoch_guard_required(&self) -> bool {
        self.exclusive_live_epoch_guard_required
    }

    pub fn under_guard_reverification_required(&self) -> bool {
        self.under_guard_reverification_required
    }

    pub fn compare_and_swap_required(&self) -> bool {
        self.compare_and_swap_required
    }

    pub fn rollback_bundle_required_before_swap(&self) -> bool {
        self.rollback_bundle_required_before_swap
    }

    pub fn post_swap_equivalence_required_before_checkpoint_commit(&self) -> bool {
        self.post_swap_equivalence_required_before_checkpoint_commit
    }

    pub fn trusted_checkpoint_commit_must_follow_installed_state_verification(&self) -> bool {
        self.trusted_checkpoint_commit_must_follow_installed_state_verification
    }

    pub fn partial_bundle_swap_allowed(&self) -> bool {
        self.partial_bundle_swap_allowed
    }

    pub fn partial_checkpoint_commit_allowed(&self) -> bool {
        self.partial_checkpoint_commit_allowed
    }

    pub fn executor_implemented(&self) -> bool {
        self.executor_implemented
    }

    pub fn live_state_lock_implemented(&self) -> bool {
        self.live_state_lock_implemented
    }

    pub fn compare_and_swap_implemented(&self) -> bool {
        self.compare_and_swap_implemented
    }

    pub fn rollback_implemented(&self) -> bool {
        self.rollback_implemented
    }

    pub fn live_state_swap_authorized(&self) -> bool {
        self.live_state_swap_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn trusted_checkpoint_commit_authorized(&self) -> bool {
        self.trusted_checkpoint_commit_authorized
    }

    pub fn contract_digest(&self) -> AtomicActivationTransactionContractDigestV1 {
        self.contract_digest
    }

    pub fn verify(&self) -> Result<(), AtomicActivationTransactionContractError> {
        let canonical = Self::canonical();
        if self.version != canonical.version {
            return Err(AtomicActivationTransactionContractError::UnsupportedVersion);
        }
        if self.required_components != canonical.required_components {
            return Err(AtomicActivationTransactionContractError::ActivationBundleMismatch);
        }
        if self.required_phases != canonical.required_phases {
            return Err(AtomicActivationTransactionContractError::PhaseOrderingMismatch);
        }
        if !self.exclusive_live_epoch_guard_required
            || !self.under_guard_reverification_required
            || !self.compare_and_swap_required
            || !self.rollback_bundle_required_before_swap
            || !self.post_swap_equivalence_required_before_checkpoint_commit
            || !self.trusted_checkpoint_commit_must_follow_installed_state_verification
            || self.partial_bundle_swap_allowed
            || self.partial_checkpoint_commit_allowed
        {
            return Err(AtomicActivationTransactionContractError::InvariantMismatch);
        }
        if self.executor_implemented
            || self.live_state_lock_implemented
            || self.compare_and_swap_implemented
            || self.rollback_implemented
            || self.live_state_swap_authorized
            || self.activation_authorized
            || self.trusted_checkpoint_commit_authorized
        {
            return Err(AtomicActivationTransactionContractError::UnexpectedAuthority);
        }
        if digest_contract(self) != self.contract_digest {
            return Err(AtomicActivationTransactionContractError::ContractDigestMismatch);
        }
        Ok(())
    }
}

fn digest_contract(
    contract: &AtomicActivationTransactionContractV1,
) -> AtomicActivationTransactionContractDigestV1 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-atomic-activation-transaction-contract-v1");
    hasher.update(&[1]);
    hasher.update(&(contract.required_components.len() as u64).to_le_bytes());
    for component in &contract.required_components {
        hasher.update(&[component.tag()]);
    }
    hasher.update(&(contract.required_phases.len() as u64).to_le_bytes());
    for phase in &contract.required_phases {
        hasher.update(&[phase.tag()]);
    }
    for value in [
        contract.exclusive_live_epoch_guard_required,
        contract.under_guard_reverification_required,
        contract.compare_and_swap_required,
        contract.rollback_bundle_required_before_swap,
        contract.post_swap_equivalence_required_before_checkpoint_commit,
        contract.trusted_checkpoint_commit_must_follow_installed_state_verification,
        contract.partial_bundle_swap_allowed,
        contract.partial_checkpoint_commit_allowed,
        contract.executor_implemented,
        contract.live_state_lock_implemented,
        contract.compare_and_swap_implemented,
        contract.rollback_implemented,
        contract.live_state_swap_authorized,
        contract.activation_authorized,
        contract.trusted_checkpoint_commit_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    AtomicActivationTransactionContractDigestV1(*hasher.finalize().as_bytes())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomicActivationTransactionContractError {
    UnsupportedVersion,
    ActivationBundleMismatch,
    PhaseOrderingMismatch,
    InvariantMismatch,
    UnexpectedAuthority,
    ContractDigestMismatch,
}

impl fmt::Display for AtomicActivationTransactionContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "atomic activation transaction contract invalid: {self:?}")
    }
}

impl Error for AtomicActivationTransactionContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_contract_is_non_authorizing_and_self_verifying() {
        let contract = AtomicActivationTransactionContractV1::canonical();
        contract.verify().unwrap();
        assert!(contract.exclusive_live_epoch_guard_required());
        assert!(contract.compare_and_swap_required());
        assert!(contract.rollback_bundle_required_before_swap());
        assert!(!contract.executor_implemented());
        assert!(!contract.live_state_swap_authorized());
        assert!(!contract.activation_authorized());
        assert!(!contract.trusted_checkpoint_commit_authorized());
    }

    #[test]
    fn phase_order_places_rollback_before_swap_and_checkpoint_commit_after_verify() {
        let contract = AtomicActivationTransactionContractV1::canonical();
        let phases = contract.required_phases();
        let rollback = phases
            .iter()
            .position(|phase| *phase == AtomicActivationPhaseV1::CaptureRollbackBundle)
            .unwrap();
        let swap = phases
            .iter()
            .position(|phase| *phase == AtomicActivationPhaseV1::AtomicLiveStateSwap)
            .unwrap();
        let verify = phases
            .iter()
            .position(|phase| *phase == AtomicActivationPhaseV1::VerifyInstalledStateUnderGuard)
            .unwrap();
        let commit = phases
            .iter()
            .position(|phase| *phase == AtomicActivationPhaseV1::CommitTrustedCheckpoints)
            .unwrap();
        assert!(rollback < swap);
        assert!(verify < commit);
    }

    #[test]
    fn complete_activation_bundle_requires_operational_revision_history() {
        let contract = AtomicActivationTransactionContractV1::canonical();
        assert!(contract
            .required_components()
            .contains(&AtomicActivationStateComponentV1::OperationalRevisionHistory));
    }
}
