// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned schema contract for epoch-bound belief-revision receipts.
//!
//! EKM-074 defines what a post-restart authority epoch must mean. This module
//! defines the receipt-schema migration required to use such an epoch safely.
//! It does not alter the current `BeliefRevisionReceipt`, create operational V2
//! receipts, or change mutation authority.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochBoundRevisionReceiptSchemaVersion {
    V2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpochBoundRevisionReceiptSchemaDigestV2([u8; 32]);

impl EpochBoundRevisionReceiptSchemaDigestV2 {
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

/// Complete semantic field inventory for the future operational receipt V2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochBoundRevisionReceiptFieldV2 {
    ReceiptId,
    ClaimId,
    ProposedDelta,
    Rationale,
    BasisEvidenceSnapshots,
    DuplicateBasisEvidenceIds,
    PolicySchema,
    CalibrationSnapshot,
    UncertaintyAssessment,
    DecisionSnapshot,
    EvaluatedAtCycle,
    AuthorityEpochDigest,
    AuthorityEpochSequence,
}

impl EpochBoundRevisionReceiptFieldV2 {
    fn tag(self) -> u8 {
        match self {
            Self::ReceiptId => 1,
            Self::ClaimId => 2,
            Self::ProposedDelta => 3,
            Self::Rationale => 4,
            Self::BasisEvidenceSnapshots => 5,
            Self::DuplicateBasisEvidenceIds => 6,
            Self::PolicySchema => 7,
            Self::CalibrationSnapshot => 8,
            Self::UncertaintyAssessment => 9,
            Self::DecisionSnapshot => 10,
            Self::EvaluatedAtCycle => 11,
            Self::AuthorityEpochDigest => 12,
            Self::AuthorityEpochSequence => 13,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochBoundRevisionReceiptMigrationRuleV2 {
    PreserveGlobalReceiptId,
    PreserveAllV1AuditSemantics,
    LegacyV1ReceiptsRemainArchivalOnly,
    EpochDigestRequiredForOperationalV2,
    EpochSequenceRequiredForOperationalV2,
    EvaluationMustNotPredateEpochActivation,
    NeverInferEpochFromCaptureCycle,
    NeverInferEpochFromRestartDigest,
    NeverInferEpochFromPreflightOrSandbox,
    NeverUpgradeLegacyV1ToOperationalV2ByInference,
    PersistEpochBinding,
    SerializeEpochBindingCanonically,
    HashEpochBindingCanonically,
    PreparedMutationMustRetainEpochBinding,
    AuthorizationMustMatchPreparedEpoch,
}

impl EpochBoundRevisionReceiptMigrationRuleV2 {
    fn tag(self) -> u8 {
        match self {
            Self::PreserveGlobalReceiptId => 1,
            Self::PreserveAllV1AuditSemantics => 2,
            Self::LegacyV1ReceiptsRemainArchivalOnly => 3,
            Self::EpochDigestRequiredForOperationalV2 => 4,
            Self::EpochSequenceRequiredForOperationalV2 => 5,
            Self::EvaluationMustNotPredateEpochActivation => 6,
            Self::NeverInferEpochFromCaptureCycle => 7,
            Self::NeverInferEpochFromRestartDigest => 8,
            Self::NeverInferEpochFromPreflightOrSandbox => 9,
            Self::NeverUpgradeLegacyV1ToOperationalV2ByInference => 10,
            Self::PersistEpochBinding => 11,
            Self::SerializeEpochBindingCanonically => 12,
            Self::HashEpochBindingCanonically => 13,
            Self::PreparedMutationMustRetainEpochBinding => 14,
            Self::AuthorizationMustMatchPreparedEpoch => 15,
        }
    }
}

/// Canonical migration contract for a future epoch-bound operational receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochBoundRevisionReceiptSchemaContractV2 {
    version: EpochBoundRevisionReceiptSchemaVersion,
    fields: Vec<EpochBoundRevisionReceiptFieldV2>,
    migration_rules: Vec<EpochBoundRevisionReceiptMigrationRuleV2>,
    global_receipt_id_continuity: bool,
    legacy_v1_readable: bool,
    legacy_v1_archival_only: bool,
    epoch_binding_mandatory_for_operational_v2: bool,
    epoch_binding_may_be_inferred: bool,
    current_receipt_type_is_v2: bool,
    current_persistence_carries_v2_epoch_binding: bool,
    current_wire_carries_v2_epoch_binding: bool,
    prepared_mutation_carries_epoch_binding: bool,
    authorization_enforces_epoch_binding: bool,
    mutation_authority_exported: bool,
    activation_authorized: bool,
    schema_digest: EpochBoundRevisionReceiptSchemaDigestV2,
}

impl EpochBoundRevisionReceiptSchemaContractV2 {
    pub fn canonical() -> Self {
        let mut out = Self {
            version: EpochBoundRevisionReceiptSchemaVersion::V2,
            fields: vec![
                EpochBoundRevisionReceiptFieldV2::ReceiptId,
                EpochBoundRevisionReceiptFieldV2::ClaimId,
                EpochBoundRevisionReceiptFieldV2::ProposedDelta,
                EpochBoundRevisionReceiptFieldV2::Rationale,
                EpochBoundRevisionReceiptFieldV2::BasisEvidenceSnapshots,
                EpochBoundRevisionReceiptFieldV2::DuplicateBasisEvidenceIds,
                EpochBoundRevisionReceiptFieldV2::PolicySchema,
                EpochBoundRevisionReceiptFieldV2::CalibrationSnapshot,
                EpochBoundRevisionReceiptFieldV2::UncertaintyAssessment,
                EpochBoundRevisionReceiptFieldV2::DecisionSnapshot,
                EpochBoundRevisionReceiptFieldV2::EvaluatedAtCycle,
                EpochBoundRevisionReceiptFieldV2::AuthorityEpochDigest,
                EpochBoundRevisionReceiptFieldV2::AuthorityEpochSequence,
            ],
            migration_rules: vec![
                EpochBoundRevisionReceiptMigrationRuleV2::PreserveGlobalReceiptId,
                EpochBoundRevisionReceiptMigrationRuleV2::PreserveAllV1AuditSemantics,
                EpochBoundRevisionReceiptMigrationRuleV2::LegacyV1ReceiptsRemainArchivalOnly,
                EpochBoundRevisionReceiptMigrationRuleV2::EpochDigestRequiredForOperationalV2,
                EpochBoundRevisionReceiptMigrationRuleV2::EpochSequenceRequiredForOperationalV2,
                EpochBoundRevisionReceiptMigrationRuleV2::EvaluationMustNotPredateEpochActivation,
                EpochBoundRevisionReceiptMigrationRuleV2::NeverInferEpochFromCaptureCycle,
                EpochBoundRevisionReceiptMigrationRuleV2::NeverInferEpochFromRestartDigest,
                EpochBoundRevisionReceiptMigrationRuleV2::NeverInferEpochFromPreflightOrSandbox,
                EpochBoundRevisionReceiptMigrationRuleV2::NeverUpgradeLegacyV1ToOperationalV2ByInference,
                EpochBoundRevisionReceiptMigrationRuleV2::PersistEpochBinding,
                EpochBoundRevisionReceiptMigrationRuleV2::SerializeEpochBindingCanonically,
                EpochBoundRevisionReceiptMigrationRuleV2::HashEpochBindingCanonically,
                EpochBoundRevisionReceiptMigrationRuleV2::PreparedMutationMustRetainEpochBinding,
                EpochBoundRevisionReceiptMigrationRuleV2::AuthorizationMustMatchPreparedEpoch,
            ],
            global_receipt_id_continuity: true,
            legacy_v1_readable: true,
            legacy_v1_archival_only: true,
            epoch_binding_mandatory_for_operational_v2: true,
            epoch_binding_may_be_inferred: false,
            current_receipt_type_is_v2: false,
            current_persistence_carries_v2_epoch_binding: false,
            current_wire_carries_v2_epoch_binding: false,
            prepared_mutation_carries_epoch_binding: false,
            authorization_enforces_epoch_binding: false,
            mutation_authority_exported: false,
            activation_authorized: false,
            schema_digest: EpochBoundRevisionReceiptSchemaDigestV2([0; 32]),
        };
        out.schema_digest = digest_contract(&out);
        out
    }

    pub fn version(&self) -> EpochBoundRevisionReceiptSchemaVersion {
        self.version
    }

    pub fn fields(&self) -> &[EpochBoundRevisionReceiptFieldV2] {
        &self.fields
    }

    pub fn migration_rules(&self) -> &[EpochBoundRevisionReceiptMigrationRuleV2] {
        &self.migration_rules
    }

    pub fn global_receipt_id_continuity(&self) -> bool {
        self.global_receipt_id_continuity
    }

    pub fn legacy_v1_readable(&self) -> bool {
        self.legacy_v1_readable
    }

    pub fn legacy_v1_archival_only(&self) -> bool {
        self.legacy_v1_archival_only
    }

    pub fn epoch_binding_mandatory_for_operational_v2(&self) -> bool {
        self.epoch_binding_mandatory_for_operational_v2
    }

    pub fn epoch_binding_may_be_inferred(&self) -> bool {
        self.epoch_binding_may_be_inferred
    }

    pub fn current_receipt_type_is_v2(&self) -> bool {
        self.current_receipt_type_is_v2
    }

    pub fn current_persistence_carries_v2_epoch_binding(&self) -> bool {
        self.current_persistence_carries_v2_epoch_binding
    }

    pub fn current_wire_carries_v2_epoch_binding(&self) -> bool {
        self.current_wire_carries_v2_epoch_binding
    }

    pub fn prepared_mutation_carries_epoch_binding(&self) -> bool {
        self.prepared_mutation_carries_epoch_binding
    }

    pub fn authorization_enforces_epoch_binding(&self) -> bool {
        self.authorization_enforces_epoch_binding
    }

    pub fn mutation_authority_exported(&self) -> bool {
        self.mutation_authority_exported
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn schema_digest(&self) -> EpochBoundRevisionReceiptSchemaDigestV2 {
        self.schema_digest
    }

    pub fn verify(&self) -> Result<(), EpochBoundRevisionReceiptSchemaError> {
        let canonical = Self::canonical();
        if self.version != canonical.version {
            return Err(EpochBoundRevisionReceiptSchemaError::UnsupportedVersion);
        }
        if self.fields != canonical.fields {
            return Err(EpochBoundRevisionReceiptSchemaError::FieldInventoryMismatch);
        }
        if self.migration_rules != canonical.migration_rules {
            return Err(EpochBoundRevisionReceiptSchemaError::MigrationRuleMismatch);
        }
        if !self.global_receipt_id_continuity
            || !self.legacy_v1_readable
            || !self.legacy_v1_archival_only
            || !self.epoch_binding_mandatory_for_operational_v2
            || self.epoch_binding_may_be_inferred
            || self.current_receipt_type_is_v2
            || self.current_persistence_carries_v2_epoch_binding
            || self.current_wire_carries_v2_epoch_binding
            || self.prepared_mutation_carries_epoch_binding
            || self.authorization_enforces_epoch_binding
            || self.mutation_authority_exported
            || self.activation_authorized
        {
            return Err(EpochBoundRevisionReceiptSchemaError::InvariantMismatch);
        }
        if digest_contract(self) != self.schema_digest {
            return Err(EpochBoundRevisionReceiptSchemaError::SchemaDigestMismatch);
        }
        Ok(())
    }
}

fn digest_contract(
    contract: &EpochBoundRevisionReceiptSchemaContractV2,
) -> EpochBoundRevisionReceiptSchemaDigestV2 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-epoch-bound-revision-receipt-schema-v2");
    hasher.update(&[2]);
    hasher.update(&(contract.fields.len() as u64).to_le_bytes());
    for field in &contract.fields {
        hasher.update(&[field.tag()]);
    }
    hasher.update(&(contract.migration_rules.len() as u64).to_le_bytes());
    for rule in &contract.migration_rules {
        hasher.update(&[rule.tag()]);
    }
    for value in [
        contract.global_receipt_id_continuity,
        contract.legacy_v1_readable,
        contract.legacy_v1_archival_only,
        contract.epoch_binding_mandatory_for_operational_v2,
        contract.epoch_binding_may_be_inferred,
        contract.current_receipt_type_is_v2,
        contract.current_persistence_carries_v2_epoch_binding,
        contract.current_wire_carries_v2_epoch_binding,
        contract.prepared_mutation_carries_epoch_binding,
        contract.authorization_enforces_epoch_binding,
        contract.mutation_authority_exported,
        contract.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    EpochBoundRevisionReceiptSchemaDigestV2(*hasher.finalize().as_bytes())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpochBoundRevisionReceiptSchemaError {
    UnsupportedVersion,
    FieldInventoryMismatch,
    MigrationRuleMismatch,
    InvariantMismatch,
    SchemaDigestMismatch,
}

impl fmt::Display for EpochBoundRevisionReceiptSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch-bound revision receipt schema invalid: {self:?}")
    }
}

impl Error for EpochBoundRevisionReceiptSchemaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_receipts_stay_readable_but_never_operationally_upgraded() {
        let contract = EpochBoundRevisionReceiptSchemaContractV2::canonical();
        contract.verify().unwrap();
        assert!(contract.legacy_v1_readable());
        assert!(contract.legacy_v1_archival_only());
        assert!(!contract.epoch_binding_may_be_inferred());
    }

    #[test]
    fn future_operational_receipts_require_explicit_epoch_binding() {
        let contract = EpochBoundRevisionReceiptSchemaContractV2::canonical();
        assert!(contract.epoch_binding_mandatory_for_operational_v2());
        assert!(contract
            .fields()
            .contains(&EpochBoundRevisionReceiptFieldV2::AuthorityEpochDigest));
        assert!(contract
            .fields()
            .contains(&EpochBoundRevisionReceiptFieldV2::AuthorityEpochSequence));
    }

    #[test]
    fn current_runtime_does_not_claim_v2_enforcement() {
        let contract = EpochBoundRevisionReceiptSchemaContractV2::canonical();
        assert!(!contract.current_receipt_type_is_v2());
        assert!(!contract.current_persistence_carries_v2_epoch_binding());
        assert!(!contract.current_wire_carries_v2_epoch_binding());
        assert!(!contract.prepared_mutation_carries_epoch_binding());
        assert!(!contract.authorization_enforces_epoch_binding());
        assert!(!contract.mutation_authority_exported());
        assert!(!contract.activation_authorized());
    }
}
