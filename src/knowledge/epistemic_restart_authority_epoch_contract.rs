// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical schema contract for post-restart epistemic authority epochs.
//!
//! EKM-073 makes all pre-restart belief-revision receipts archival-only. This
//! module defines what a *new* authority epoch must mean before fresh receipts
//! can regain mutation authority. It intentionally does not mint an epoch, alter
//! [`BeliefRevisionReceipt`], or change [`BeliefMutationAuthority`].
//!
//! An operational epoch may only be issued by a future successful atomic
//! activation commit. Preflight, review eligibility, a sandbox, or a digest alone
//! are insufficient to create post-restart mutation authority.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartAuthorityEpochContractVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartAuthorityEpochContractDigestV1([u8; 32]);

impl RestartAuthorityEpochContractDigestV1 {
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

/// Fields that a future operational restart-authority epoch identity must bind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartAuthorityEpochIdentityFieldV1 {
    DeploymentId,
    TrustDomainId,
    EpochSequence,
    PreviousEpochDigest,
    ActivatedRestartV2Digest,
    ActivationCommitReceiptDigest,
    ActivatedLiveGeneration,
    ActivatedAtCycle,
}

impl RestartAuthorityEpochIdentityFieldV1 {
    fn tag(self) -> u8 {
        match self {
            Self::DeploymentId => 1,
            Self::TrustDomainId => 2,
            Self::EpochSequence => 3,
            Self::PreviousEpochDigest => 4,
            Self::ActivatedRestartV2Digest => 5,
            Self::ActivationCommitReceiptDigest => 6,
            Self::ActivatedLiveGeneration => 7,
            Self::ActivatedAtCycle => 8,
        }
    }
}

/// Rules that a future epoch-bound belief-revision receipt schema must enforce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartAuthorityEpochReceiptRequirementV1 {
    PreserveGlobalReceiptIdContinuity,
    BindReceiptToEpochDigest,
    BindReceiptToEpochSequence,
    EvaluationMustNotPredateEpochActivation,
    PreparationMustUseActiveEpoch,
    AuthorizationMustMatchPreparedReceiptEpoch,
    PersistEpochBinding,
    SerializeEpochBindingCanonically,
    HistoricalEpochReceiptsRemainArchival,
}

impl RestartAuthorityEpochReceiptRequirementV1 {
    fn tag(self) -> u8 {
        match self {
            Self::PreserveGlobalReceiptIdContinuity => 1,
            Self::BindReceiptToEpochDigest => 2,
            Self::BindReceiptToEpochSequence => 3,
            Self::EvaluationMustNotPredateEpochActivation => 4,
            Self::PreparationMustUseActiveEpoch => 5,
            Self::AuthorizationMustMatchPreparedReceiptEpoch => 6,
            Self::PersistEpochBinding => 7,
            Self::SerializeEpochBindingCanonically => 8,
            Self::HistoricalEpochReceiptsRemainArchival => 9,
        }
    }
}

/// Candidate-independent V1 contract for restart authority epochs.
///
/// This is a schema/authority specification only. The flags that would imply an
/// implemented authority path are deliberately false until later qualified work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartAuthorityEpochContractV1 {
    version: RestartAuthorityEpochContractVersion,
    identity_fields: Vec<RestartAuthorityEpochIdentityFieldV1>,
    receipt_requirements: Vec<RestartAuthorityEpochReceiptRequirementV1>,
    epoch_issuance_requires_successful_activation_commit: bool,
    preflight_may_issue_epoch: bool,
    review_eligibility_may_issue_epoch: bool,
    sandbox_may_issue_epoch: bool,
    receipt_ids_global_across_epochs: bool,
    historical_receipts_archival_only: bool,
    current_receipt_schema_has_epoch_binding: bool,
    current_mutation_facade_enforces_active_epoch: bool,
    operational_epoch_issuance_implemented: bool,
    operational_history_continuation_implemented: bool,
    mutation_authority_exported: bool,
    activation_authorized: bool,
    contract_digest: RestartAuthorityEpochContractDigestV1,
}

impl RestartAuthorityEpochContractV1 {
    pub fn canonical() -> Self {
        let mut out = Self {
            version: RestartAuthorityEpochContractVersion::V1,
            identity_fields: vec![
                RestartAuthorityEpochIdentityFieldV1::DeploymentId,
                RestartAuthorityEpochIdentityFieldV1::TrustDomainId,
                RestartAuthorityEpochIdentityFieldV1::EpochSequence,
                RestartAuthorityEpochIdentityFieldV1::PreviousEpochDigest,
                RestartAuthorityEpochIdentityFieldV1::ActivatedRestartV2Digest,
                RestartAuthorityEpochIdentityFieldV1::ActivationCommitReceiptDigest,
                RestartAuthorityEpochIdentityFieldV1::ActivatedLiveGeneration,
                RestartAuthorityEpochIdentityFieldV1::ActivatedAtCycle,
            ],
            receipt_requirements: vec![
                RestartAuthorityEpochReceiptRequirementV1::PreserveGlobalReceiptIdContinuity,
                RestartAuthorityEpochReceiptRequirementV1::BindReceiptToEpochDigest,
                RestartAuthorityEpochReceiptRequirementV1::BindReceiptToEpochSequence,
                RestartAuthorityEpochReceiptRequirementV1::EvaluationMustNotPredateEpochActivation,
                RestartAuthorityEpochReceiptRequirementV1::PreparationMustUseActiveEpoch,
                RestartAuthorityEpochReceiptRequirementV1::AuthorizationMustMatchPreparedReceiptEpoch,
                RestartAuthorityEpochReceiptRequirementV1::PersistEpochBinding,
                RestartAuthorityEpochReceiptRequirementV1::SerializeEpochBindingCanonically,
                RestartAuthorityEpochReceiptRequirementV1::HistoricalEpochReceiptsRemainArchival,
            ],
            epoch_issuance_requires_successful_activation_commit: true,
            preflight_may_issue_epoch: false,
            review_eligibility_may_issue_epoch: false,
            sandbox_may_issue_epoch: false,
            receipt_ids_global_across_epochs: true,
            historical_receipts_archival_only: true,
            current_receipt_schema_has_epoch_binding: false,
            current_mutation_facade_enforces_active_epoch: false,
            operational_epoch_issuance_implemented: false,
            operational_history_continuation_implemented: false,
            mutation_authority_exported: false,
            activation_authorized: false,
            contract_digest: RestartAuthorityEpochContractDigestV1([0; 32]),
        };
        out.contract_digest = digest_contract(&out);
        out
    }

    pub fn version(&self) -> RestartAuthorityEpochContractVersion {
        self.version
    }

    pub fn identity_fields(&self) -> &[RestartAuthorityEpochIdentityFieldV1] {
        &self.identity_fields
    }

    pub fn receipt_requirements(&self) -> &[RestartAuthorityEpochReceiptRequirementV1] {
        &self.receipt_requirements
    }

    pub fn epoch_issuance_requires_successful_activation_commit(&self) -> bool {
        self.epoch_issuance_requires_successful_activation_commit
    }

    pub fn preflight_may_issue_epoch(&self) -> bool {
        self.preflight_may_issue_epoch
    }

    pub fn review_eligibility_may_issue_epoch(&self) -> bool {
        self.review_eligibility_may_issue_epoch
    }

    pub fn sandbox_may_issue_epoch(&self) -> bool {
        self.sandbox_may_issue_epoch
    }

    pub fn receipt_ids_global_across_epochs(&self) -> bool {
        self.receipt_ids_global_across_epochs
    }

    pub fn historical_receipts_archival_only(&self) -> bool {
        self.historical_receipts_archival_only
    }

    pub fn current_receipt_schema_has_epoch_binding(&self) -> bool {
        self.current_receipt_schema_has_epoch_binding
    }

    pub fn current_mutation_facade_enforces_active_epoch(&self) -> bool {
        self.current_mutation_facade_enforces_active_epoch
    }

    pub fn operational_epoch_issuance_implemented(&self) -> bool {
        self.operational_epoch_issuance_implemented
    }

    pub fn operational_history_continuation_implemented(&self) -> bool {
        self.operational_history_continuation_implemented
    }

    pub fn mutation_authority_exported(&self) -> bool {
        self.mutation_authority_exported
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn contract_digest(&self) -> RestartAuthorityEpochContractDigestV1 {
        self.contract_digest
    }

    pub fn verify(&self) -> Result<(), RestartAuthorityEpochContractError> {
        let canonical = Self::canonical();
        if self.version != canonical.version {
            return Err(RestartAuthorityEpochContractError::UnsupportedVersion);
        }
        if self.identity_fields != canonical.identity_fields {
            return Err(RestartAuthorityEpochContractError::IdentityFieldMismatch);
        }
        if self.receipt_requirements != canonical.receipt_requirements {
            return Err(RestartAuthorityEpochContractError::ReceiptRequirementMismatch);
        }
        if !self.epoch_issuance_requires_successful_activation_commit
            || self.preflight_may_issue_epoch
            || self.review_eligibility_may_issue_epoch
            || self.sandbox_may_issue_epoch
            || !self.receipt_ids_global_across_epochs
            || !self.historical_receipts_archival_only
            || self.current_receipt_schema_has_epoch_binding
            || self.current_mutation_facade_enforces_active_epoch
            || self.operational_epoch_issuance_implemented
            || self.operational_history_continuation_implemented
            || self.mutation_authority_exported
            || self.activation_authorized
        {
            return Err(RestartAuthorityEpochContractError::InvariantMismatch);
        }
        if digest_contract(self) != self.contract_digest {
            return Err(RestartAuthorityEpochContractError::ContractDigestMismatch);
        }
        Ok(())
    }
}

fn digest_contract(contract: &RestartAuthorityEpochContractV1) -> RestartAuthorityEpochContractDigestV1 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-authority-epoch-contract-v1");
    hasher.update(&[1]);
    hasher.update(&(contract.identity_fields.len() as u64).to_le_bytes());
    for field in &contract.identity_fields {
        hasher.update(&[field.tag()]);
    }
    hasher.update(&(contract.receipt_requirements.len() as u64).to_le_bytes());
    for requirement in &contract.receipt_requirements {
        hasher.update(&[requirement.tag()]);
    }
    for value in [
        contract.epoch_issuance_requires_successful_activation_commit,
        contract.preflight_may_issue_epoch,
        contract.review_eligibility_may_issue_epoch,
        contract.sandbox_may_issue_epoch,
        contract.receipt_ids_global_across_epochs,
        contract.historical_receipts_archival_only,
        contract.current_receipt_schema_has_epoch_binding,
        contract.current_mutation_facade_enforces_active_epoch,
        contract.operational_epoch_issuance_implemented,
        contract.operational_history_continuation_implemented,
        contract.mutation_authority_exported,
        contract.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    RestartAuthorityEpochContractDigestV1(*hasher.finalize().as_bytes())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartAuthorityEpochContractError {
    UnsupportedVersion,
    IdentityFieldMismatch,
    ReceiptRequirementMismatch,
    InvariantMismatch,
    ContractDigestMismatch,
}

impl fmt::Display for RestartAuthorityEpochContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart authority epoch contract invalid: {self:?}")
    }
}

impl Error for RestartAuthorityEpochContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_commit_is_the_only_allowed_epoch_issuance_boundary() {
        let contract = RestartAuthorityEpochContractV1::canonical();
        contract.verify().unwrap();
        assert!(contract.epoch_issuance_requires_successful_activation_commit());
        assert!(!contract.preflight_may_issue_epoch());
        assert!(!contract.review_eligibility_may_issue_epoch());
        assert!(!contract.sandbox_may_issue_epoch());
    }

    #[test]
    fn receipt_ids_continue_globally_but_authority_is_epoch_scoped() {
        let contract = RestartAuthorityEpochContractV1::canonical();
        assert!(contract.receipt_ids_global_across_epochs());
        assert!(contract.historical_receipts_archival_only());
        assert!(contract
            .receipt_requirements()
            .contains(&RestartAuthorityEpochReceiptRequirementV1::BindReceiptToEpochDigest));
        assert!(contract.receipt_requirements().contains(
            &RestartAuthorityEpochReceiptRequirementV1::PreparationMustUseActiveEpoch
        ));
    }

    #[test]
    fn current_schema_and_facade_do_not_claim_epoch_enforcement() {
        let contract = RestartAuthorityEpochContractV1::canonical();
        assert!(!contract.current_receipt_schema_has_epoch_binding());
        assert!(!contract.current_mutation_facade_enforces_active_epoch());
        assert!(!contract.operational_epoch_issuance_implemented());
        assert!(!contract.operational_history_continuation_implemented());
        assert!(!contract.mutation_authority_exported());
        assert!(!contract.activation_authorized());
    }
}
