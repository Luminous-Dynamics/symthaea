// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority contract for revision receipts across restart boundaries.
//!
//! Historical revision receipts are valuable audit lineage, but restoring them as
//! fresh mutation authority would be unsafe. Applied receipts have already had
//! their effects replay-verified into support state; rejected receipts never had
//! authority; and eligible-but-unapplied receipts may be stale after restart.
//!
//! This module therefore defines a candidate-independent restart rule: historical
//! receipts are archival only, and every post-restart mutation must be authorized
//! by a freshly evaluated receipt bound to the new restart epoch. No operational
//! history restore or mutation path is implemented here.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartRevisionAuthorityContractVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartRevisionAuthorityContractDigestV1([u8; 32]);

impl RestartRevisionAuthorityContractDigestV1 {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalRevisionReceiptClassV1 {
    Applied,
    Rejected,
    EligibleUnapplied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalRevisionAuthorityDispositionV1 {
    /// Receipt is preserved for audit/identity continuity only.
    ArchivalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartRevisionAuthorityRequirementV1 {
    PreserveHistoricalReceiptIdentity,
    PreserveNextReceiptIdContinuity,
    ArchiveHistoricalAppliedReceipts,
    ArchiveHistoricalRejectedReceipts,
    ArchiveHistoricalEligibleUnappliedReceipts,
    ForbidHistoricalAuthorizationReplay,
    RequireFreshPostRestartEvaluation,
    BindFreshReceiptsToRestartEpoch,
    PreserveHistoricalAppliedMutationLinkage,
}

impl RestartRevisionAuthorityRequirementV1 {
    fn tag(self) -> u8 {
        match self {
            Self::PreserveHistoricalReceiptIdentity => 1,
            Self::PreserveNextReceiptIdContinuity => 2,
            Self::ArchiveHistoricalAppliedReceipts => 3,
            Self::ArchiveHistoricalRejectedReceipts => 4,
            Self::ArchiveHistoricalEligibleUnappliedReceipts => 5,
            Self::ForbidHistoricalAuthorizationReplay => 6,
            Self::RequireFreshPostRestartEvaluation => 7,
            Self::BindFreshReceiptsToRestartEpoch => 8,
            Self::PreserveHistoricalAppliedMutationLinkage => 9,
        }
    }
}

/// Canonical restart authority policy for belief-revision receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartRevisionAuthorityContractV1 {
    version: RestartRevisionAuthorityContractVersion,
    requirements: Vec<RestartRevisionAuthorityRequirementV1>,
    historical_applied_disposition: HistoricalRevisionAuthorityDispositionV1,
    historical_rejected_disposition: HistoricalRevisionAuthorityDispositionV1,
    historical_eligible_unapplied_disposition: HistoricalRevisionAuthorityDispositionV1,
    historical_receipts_mutation_authority: bool,
    applied_authorization_replay_allowed: bool,
    eligible_unapplied_authorization_replay_allowed: bool,
    next_receipt_id_continuity_required: bool,
    fresh_post_restart_evaluation_required: bool,
    restart_epoch_binding_required: bool,
    current_receipt_schema_has_restart_epoch_binding: bool,
    operational_history_restore_implemented: bool,
    mutation_authority_exported: bool,
    activation_authorized: bool,
    contract_digest: RestartRevisionAuthorityContractDigestV1,
}

impl RestartRevisionAuthorityContractV1 {
    pub fn canonical() -> Self {
        let mut out = Self {
            version: RestartRevisionAuthorityContractVersion::V1,
            requirements: vec![
                RestartRevisionAuthorityRequirementV1::PreserveHistoricalReceiptIdentity,
                RestartRevisionAuthorityRequirementV1::PreserveNextReceiptIdContinuity,
                RestartRevisionAuthorityRequirementV1::ArchiveHistoricalAppliedReceipts,
                RestartRevisionAuthorityRequirementV1::ArchiveHistoricalRejectedReceipts,
                RestartRevisionAuthorityRequirementV1::ArchiveHistoricalEligibleUnappliedReceipts,
                RestartRevisionAuthorityRequirementV1::ForbidHistoricalAuthorizationReplay,
                RestartRevisionAuthorityRequirementV1::RequireFreshPostRestartEvaluation,
                RestartRevisionAuthorityRequirementV1::BindFreshReceiptsToRestartEpoch,
                RestartRevisionAuthorityRequirementV1::PreserveHistoricalAppliedMutationLinkage,
            ],
            historical_applied_disposition: HistoricalRevisionAuthorityDispositionV1::ArchivalOnly,
            historical_rejected_disposition: HistoricalRevisionAuthorityDispositionV1::ArchivalOnly,
            historical_eligible_unapplied_disposition:
                HistoricalRevisionAuthorityDispositionV1::ArchivalOnly,
            historical_receipts_mutation_authority: false,
            applied_authorization_replay_allowed: false,
            eligible_unapplied_authorization_replay_allowed: false,
            next_receipt_id_continuity_required: true,
            fresh_post_restart_evaluation_required: true,
            restart_epoch_binding_required: true,
            current_receipt_schema_has_restart_epoch_binding: false,
            operational_history_restore_implemented: false,
            mutation_authority_exported: false,
            activation_authorized: false,
            contract_digest: RestartRevisionAuthorityContractDigestV1([0; 32]),
        };
        out.contract_digest = digest_contract(&out);
        out
    }

    pub fn version(&self) -> RestartRevisionAuthorityContractVersion {
        self.version
    }

    pub fn requirements(&self) -> &[RestartRevisionAuthorityRequirementV1] {
        &self.requirements
    }

    pub fn disposition_for(
        &self,
        _class: HistoricalRevisionReceiptClassV1,
    ) -> HistoricalRevisionAuthorityDispositionV1 {
        HistoricalRevisionAuthorityDispositionV1::ArchivalOnly
    }

    pub fn historical_receipts_mutation_authority(&self) -> bool {
        self.historical_receipts_mutation_authority
    }

    pub fn applied_authorization_replay_allowed(&self) -> bool {
        self.applied_authorization_replay_allowed
    }

    pub fn eligible_unapplied_authorization_replay_allowed(&self) -> bool {
        self.eligible_unapplied_authorization_replay_allowed
    }

    pub fn next_receipt_id_continuity_required(&self) -> bool {
        self.next_receipt_id_continuity_required
    }

    pub fn fresh_post_restart_evaluation_required(&self) -> bool {
        self.fresh_post_restart_evaluation_required
    }

    pub fn restart_epoch_binding_required(&self) -> bool {
        self.restart_epoch_binding_required
    }

    pub fn current_receipt_schema_has_restart_epoch_binding(&self) -> bool {
        self.current_receipt_schema_has_restart_epoch_binding
    }

    pub fn operational_history_restore_implemented(&self) -> bool {
        self.operational_history_restore_implemented
    }

    pub fn mutation_authority_exported(&self) -> bool {
        self.mutation_authority_exported
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn contract_digest(&self) -> RestartRevisionAuthorityContractDigestV1 {
        self.contract_digest
    }

    pub fn verify(&self) -> Result<(), RestartRevisionAuthorityContractError> {
        let canonical = Self::canonical();
        if self.version != canonical.version {
            return Err(RestartRevisionAuthorityContractError::UnsupportedVersion);
        }
        if self.requirements != canonical.requirements {
            return Err(RestartRevisionAuthorityContractError::RequirementMismatch);
        }
        if self.historical_applied_disposition
            != HistoricalRevisionAuthorityDispositionV1::ArchivalOnly
            || self.historical_rejected_disposition
                != HistoricalRevisionAuthorityDispositionV1::ArchivalOnly
            || self.historical_eligible_unapplied_disposition
                != HistoricalRevisionAuthorityDispositionV1::ArchivalOnly
        {
            return Err(RestartRevisionAuthorityContractError::HistoricalAuthorityMismatch);
        }
        if self.historical_receipts_mutation_authority
            || self.applied_authorization_replay_allowed
            || self.eligible_unapplied_authorization_replay_allowed
            || !self.next_receipt_id_continuity_required
            || !self.fresh_post_restart_evaluation_required
            || !self.restart_epoch_binding_required
            || self.current_receipt_schema_has_restart_epoch_binding
            || self.operational_history_restore_implemented
            || self.mutation_authority_exported
            || self.activation_authorized
        {
            return Err(RestartRevisionAuthorityContractError::InvariantMismatch);
        }
        if digest_contract(self) != self.contract_digest {
            return Err(RestartRevisionAuthorityContractError::ContractDigestMismatch);
        }
        Ok(())
    }
}

fn digest_contract(
    contract: &RestartRevisionAuthorityContractV1,
) -> RestartRevisionAuthorityContractDigestV1 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-revision-authority-contract-v1");
    hasher.update(&[1]);
    hasher.update(&(contract.requirements.len() as u64).to_le_bytes());
    for requirement in &contract.requirements {
        hasher.update(&[requirement.tag()]);
    }
    for value in [
        contract.historical_receipts_mutation_authority,
        contract.applied_authorization_replay_allowed,
        contract.eligible_unapplied_authorization_replay_allowed,
        contract.next_receipt_id_continuity_required,
        contract.fresh_post_restart_evaluation_required,
        contract.restart_epoch_binding_required,
        contract.current_receipt_schema_has_restart_epoch_binding,
        contract.operational_history_restore_implemented,
        contract.mutation_authority_exported,
        contract.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    RestartRevisionAuthorityContractDigestV1(*hasher.finalize().as_bytes())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartRevisionAuthorityContractError {
    UnsupportedVersion,
    RequirementMismatch,
    HistoricalAuthorityMismatch,
    InvariantMismatch,
    ContractDigestMismatch,
}

impl fmt::Display for RestartRevisionAuthorityContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart revision authority contract invalid: {self:?}")
    }
}

impl Error for RestartRevisionAuthorityContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_historical_receipt_class_is_archival_only() {
        let contract = RestartRevisionAuthorityContractV1::canonical();
        contract.verify().unwrap();
        for class in [
            HistoricalRevisionReceiptClassV1::Applied,
            HistoricalRevisionReceiptClassV1::Rejected,
            HistoricalRevisionReceiptClassV1::EligibleUnapplied,
        ] {
            assert_eq!(
                contract.disposition_for(class),
                HistoricalRevisionAuthorityDispositionV1::ArchivalOnly
            );
        }
        assert!(!contract.historical_receipts_mutation_authority());
    }

    #[test]
    fn stale_eligible_receipts_cannot_be_replayed_after_restart() {
        let contract = RestartRevisionAuthorityContractV1::canonical();
        assert!(!contract.eligible_unapplied_authorization_replay_allowed());
        assert!(contract.fresh_post_restart_evaluation_required());
        assert!(contract.restart_epoch_binding_required());
    }

    #[test]
    fn epoch_binding_is_required_but_not_yet_present() {
        let contract = RestartRevisionAuthorityContractV1::canonical();
        assert!(contract.restart_epoch_binding_required());
        assert!(!contract.current_receipt_schema_has_restart_epoch_binding());
        assert!(!contract.operational_history_restore_implemented());
        assert!(!contract.activation_authorized());
    }
}
