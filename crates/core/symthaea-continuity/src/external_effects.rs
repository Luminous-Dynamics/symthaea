// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit external-side-effect obligations and closed-world reconciliation.
//!
//! Returning a server, switch, storage controller, or service to source realization A
//! does not imply that externally visible effects from the attempted A -> B transition
//! disappeared. This module makes those effects explicit and independently observable.
//!
//! The scope is deliberately honest: qualification proves every obligation in one
//! exact coverage manifest is reconciled. It never claims that no undeclared effect
//! exists elsewhere.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution_capability::ExecutionAttemptId;
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodExecutionIntentId,
    KnownGoodTransitionLineageError, KnownGoodTransitionLineageId,
};
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::TargetRealizationId;

pub const EXTERNAL_EFFECT_CONTRACT_SCHEMA_V1: &str =
    "symthaea-continuity-external-effect-contract-v1";
pub const EXTERNAL_EFFECT_OBSERVATION_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-external-effect-observation-claim-v1";
pub const EXTERNAL_EFFECT_OBSERVATION_AUTH_PURPOSE: &str =
    "symthaea.continuity.external-effect-observation.v1";
pub const EXTERNAL_EFFECT_RECONCILIATION_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-external-effect-reconciliation-record-v1";

const OBLIGATION_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-obligation.v1\0";
const CONTRACT_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-contract.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-observation-policy.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-observation-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-observation-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-external-effect-observation.v1\0";
const QUALIFIED_OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-external-effect-observation.v1\0";
const RECONCILIATION_DOMAIN: &[u8] =
    b"symthaea.continuity.external-effect-reconciliation.v1\0";
const MAX_CUSTOM_KIND_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExternalEffectObligationId([u8; 32]);
impl ExternalEffectObligationId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExternalEffectContractId([u8; 32]);
impl ExternalEffectContractId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExternalEffectObservationPolicyId([u8; 32]);
impl ExternalEffectObservationPolicyId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExternalEffectObservationClaimId([u8; 32]);
impl ExternalEffectObservationClaimId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedExternalEffectObservationId([u8; 32]);
impl AuthenticatedExternalEffectObservationId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedExternalEffectObservationId([u8; 32]);
impl QualifiedExternalEffectObservationId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedExternalEffectReconciliationId([u8; 32]);
impl QualifiedExternalEffectReconciliationId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

/// Broad external effect classes. `Custom` permits protocol-specific extensions while
/// preserving a canonical class identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalEffectClassV1 {
    DatabaseMutation,
    MessageEmission,
    ExternalApiMutation,
    NetworkControlPlaneMutation,
    StorageMutation,
    IdentityOrCredentialMutation,
    DeviceActuation,
    Custom { kind_id: String },
}

/// What must be independently established before one declared effect is considered
/// reconciled with source-state recovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalEffectRecoveryPredicateV1 {
    MustBeAbsent,
    MustMatchSourceState { source_state_digest: [u8; 32] },
    MustBeCompensated { compensation_contract_digest: [u8; 32] },
    MayPersistUnderPolicy { acceptance_policy_digest: [u8; 32] },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEffectObligationV1 {
    effect_class: ExternalEffectClassV1,
    destination_scope_digest: [u8; 32],
    operation_digest: [u8; 32],
    idempotency_key_digest: Option<[u8; 32]>,
    recovery_predicate: ExternalEffectRecoveryPredicateV1,
    obligation_id: ExternalEffectObligationId,
}

impl ExternalEffectObligationV1 {
    pub fn new(
        effect_class: ExternalEffectClassV1,
        destination_scope_digest: [u8; 32],
        operation_digest: [u8; 32],
        idempotency_key_digest: Option<[u8; 32]>,
        recovery_predicate: ExternalEffectRecoveryPredicateV1,
    ) -> Result<Self, ExternalEffectError> {
        validate_class(&effect_class)?;
        validate_obligation_material(
            destination_scope_digest,
            operation_digest,
            idempotency_key_digest,
            &recovery_predicate,
        )?;
        let obligation_id = ExternalEffectObligationId(hash_obligation(
            &effect_class,
            destination_scope_digest,
            operation_digest,
            idempotency_key_digest,
            &recovery_predicate,
        ));
        Ok(Self {
            effect_class,
            destination_scope_digest,
            operation_digest,
            idempotency_key_digest,
            recovery_predicate,
            obligation_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExternalEffectError> {
        validate_class(&self.effect_class)?;
        validate_obligation_material(
            self.destination_scope_digest,
            self.operation_digest,
            self.idempotency_key_digest,
            &self.recovery_predicate,
        )?;
        let expected = ExternalEffectObligationId(hash_obligation(
            &self.effect_class,
            self.destination_scope_digest,
            self.operation_digest,
            self.idempotency_key_digest,
            &self.recovery_predicate,
        ));
        if expected != self.obligation_id {
            return Err(ExternalEffectError::ObligationIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ExternalEffectObligationId { self.obligation_id }
    pub fn effect_class(&self) -> &ExternalEffectClassV1 { &self.effect_class }
    pub fn recovery_predicate(&self) -> &ExternalEffectRecoveryPredicateV1 {
        &self.recovery_predicate
    }
}

/// Pre-execution external-effect boundary bound to one exact durable A -> B intent.
/// `coverage_manifest_digest` identifies the adapter/system analysis that defined the
/// claimed external world. This contract does not prove that analysis exhaustive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEffectContractV1 {
    schema_version: String,
    known_good_intent_id: KnownGoodExecutionIntentId,
    transition_lineage_id: KnownGoodTransitionLineageId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    coverage_manifest_digest: [u8; 32],
    obligations: Vec<ExternalEffectObligationV1>,
    contract_id: ExternalEffectContractId,
}

impl ExternalEffectContractV1 {
    pub fn new(
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        coverage_manifest_digest: [u8; 32],
        mut obligations: Vec<ExternalEffectObligationV1>,
    ) -> Result<Self, ExternalEffectError> {
        intent.validate()?;
        if coverage_manifest_digest == [0; 32] {
            return Err(ExternalEffectError::ZeroCoverageManifestDigest);
        }
        for obligation in &obligations { obligation.validate()?; }
        obligations.sort_by_key(ExternalEffectObligationV1::id);
        if obligations.windows(2).any(|pair| pair[0].id() == pair[1].id()) {
            return Err(ExternalEffectError::DuplicateObligation);
        }
        let lineage = intent.lineage();
        let contract_id = ExternalEffectContractId(hash_contract(
            intent.id(),
            lineage.id(),
            intent.attempt_id(),
            lineage.subject_id(),
            lineage.source_realization_id(),
            lineage.target_realization_id(),
            coverage_manifest_digest,
            &obligations,
        ));
        Ok(Self {
            schema_version: EXTERNAL_EFFECT_CONTRACT_SCHEMA_V1.to_owned(),
            known_good_intent_id: intent.id(),
            transition_lineage_id: lineage.id(),
            attempt_id: intent.attempt_id(),
            subject_id: lineage.subject_id(),
            source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(),
            coverage_manifest_digest,
            obligations,
            contract_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExternalEffectError> {
        if self.schema_version != EXTERNAL_EFFECT_CONTRACT_SCHEMA_V1 {
            return Err(ExternalEffectError::UnsupportedContractSchema(self.schema_version.clone()));
        }
        if self.source_realization_id == self.target_realization_id {
            return Err(ExternalEffectError::SourceEqualsTarget);
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(ExternalEffectError::ZeroCoverageManifestDigest);
        }
        for obligation in &self.obligations { obligation.validate()?; }
        if self.obligations.windows(2).any(|pair| pair[0].id() >= pair[1].id()) {
            return Err(ExternalEffectError::NonCanonicalObligations);
        }
        let expected = ExternalEffectContractId(hash_contract(
            self.known_good_intent_id,
            self.transition_lineage_id,
            self.attempt_id,
            self.subject_id,
            self.source_realization_id,
            self.target_realization_id,
            self.coverage_manifest_digest,
            &self.obligations,
        ));
        if expected != self.contract_id {
            return Err(ExternalEffectError::ContractIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ExternalEffectContractId { self.contract_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn obligations(&self) -> &[ExternalEffectObligationV1] { &self.obligations }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalEffectObservationPolicyV1 {
    policy_id: ExternalEffectObservationPolicyId,
    generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    maximum_observation_age_ms: u64,
    maximum_future_skew_ms: u64,
}

impl ExternalEffectObservationPolicyV1 {
    pub fn new(
        profile: &VerifierProfileV1,
        generation: u64,
        maximum_observation_age_ms: u64,
        maximum_future_skew_ms: u64,
    ) -> Result<Self, ExternalEffectError> {
        profile.validate()?;
        if generation == 0 { return Err(ExternalEffectError::ZeroPolicyGeneration); }
        if maximum_observation_age_ms == 0 {
            return Err(ExternalEffectError::ZeroMaximumObservationAge);
        }
        let policy_id = ExternalEffectObservationPolicyId(domain_hash_parts(
            POLICY_DOMAIN,
            &[
                profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(),
                &generation.to_le_bytes(),
                &maximum_observation_age_ms.to_le_bytes(),
                &maximum_future_skew_ms.to_le_bytes(),
            ],
        ));
        Ok(Self {
            policy_id,
            generation,
            verifier_profile_id: profile.id(),
            verifier_root_epoch: profile.root_epoch(),
            maximum_observation_age_ms,
            maximum_future_skew_ms,
        })
    }

    pub fn validate_against_profile(&self, profile: &VerifierProfileV1) -> Result<(), ExternalEffectError> {
        profile.validate()?;
        if self.generation == 0 || self.maximum_observation_age_ms == 0 {
            return Err(ExternalEffectError::PolicyIdentityMismatch);
        }
        if profile.id() != self.verifier_profile_id || profile.root_epoch() != self.verifier_root_epoch {
            return Err(ExternalEffectError::VerifierProfileMismatch);
        }
        let expected = ExternalEffectObservationPolicyId(domain_hash_parts(
            POLICY_DOMAIN,
            &[
                self.verifier_profile_id.as_bytes(),
                &self.verifier_root_epoch.to_le_bytes(),
                &self.generation.to_le_bytes(),
                &self.maximum_observation_age_ms.to_le_bytes(),
                &self.maximum_future_skew_ms.to_le_bytes(),
            ],
        ));
        if expected != self.policy_id { return Err(ExternalEffectError::PolicyIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> ExternalEffectObservationPolicyId { self.policy_id }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalEffectObservedStateV1 {
    Absent,
    SourceEquivalent,
    Compensated,
    AcceptedPersistent,
    PresentUnreconciled,
    Unknown,
}

impl ExternalEffectObservedStateV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Absent => 1,
            Self::SourceEquivalent => 2,
            Self::Compensated => 3,
            Self::AcceptedPersistent => 4,
            Self::PresentUnreconciled => 5,
            Self::Unknown => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEffectObservationClaimV1 {
    schema_version: String,
    contract_id: ExternalEffectContractId,
    obligation_id: ExternalEffectObligationId,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    observed_state: ExternalEffectObservedStateV1,
    observed_state_digest: Option<[u8; 32]>,
    resolution_basis_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: ExternalEffectObservationClaimId,
}

impl ExternalEffectObservationClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        contract: &ExternalEffectContractV1,
        obligation: &ExternalEffectObligationV1,
        verifier_profile_id: VerifierProfileId,
        observed_at_unix_ms: u64,
        observed_state: ExternalEffectObservedStateV1,
        observed_state_digest: Option<[u8; 32]>,
        resolution_basis_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, ExternalEffectError> {
        contract.validate()?;
        obligation.validate()?;
        if contract.obligations.binary_search_by_key(&obligation.id(), ExternalEffectObligationV1::id).is_err() {
            return Err(ExternalEffectError::ObligationOutsideContract);
        }
        validate_observation_material(
            observed_at_unix_ms,
            observed_state,
            observed_state_digest,
            resolution_basis_digest,
            raw_evidence_digest,
        )?;
        let claim_id = ExternalEffectObservationClaimId(hash_observation_claim(
            contract.id(),
            obligation.id(),
            verifier_profile_id,
            observed_at_unix_ms,
            observed_state,
            observed_state_digest,
            resolution_basis_digest,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: EXTERNAL_EFFECT_OBSERVATION_CLAIM_SCHEMA_V1.to_owned(),
            contract_id: contract.id(),
            obligation_id: obligation.id(),
            verifier_profile_id,
            observed_at_unix_ms,
            observed_state,
            observed_state_digest,
            resolution_basis_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExternalEffectError> {
        if self.schema_version != EXTERNAL_EFFECT_OBSERVATION_CLAIM_SCHEMA_V1 {
            return Err(ExternalEffectError::UnsupportedObservationSchema(self.schema_version.clone()));
        }
        validate_observation_material(
            self.observed_at_unix_ms,
            self.observed_state,
            self.observed_state_digest,
            self.resolution_basis_digest,
            self.raw_evidence_digest,
        )?;
        let expected = ExternalEffectObservationClaimId(hash_observation_claim(
            self.contract_id,
            self.obligation_id,
            self.verifier_profile_id,
            self.observed_at_unix_ms,
            self.observed_state,
            self.observed_state_digest,
            self.resolution_basis_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id { return Err(ExternalEffectError::ObservationIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> ExternalEffectObservationClaimId { self.claim_id }
}

pub fn canonical_external_effect_observation_claim_bytes(
    claim: &ExternalEffectObservationClaimV1,
) -> Result<Vec<u8>, ExternalEffectError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(384);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.contract_id.as_bytes());
    out.extend_from_slice(claim.obligation_id.as_bytes());
    out.extend_from_slice(claim.verifier_profile_id.as_bytes());
    out.extend_from_slice(&claim.observed_at_unix_ms.to_le_bytes());
    out.push(claim.observed_state.tag());
    encode_optional_digest(&mut out, claim.observed_state_digest);
    encode_optional_digest(&mut out, claim.resolution_basis_digest);
    out.extend_from_slice(&claim.raw_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_external_effect_observation_claim_digest(
    claim: &ExternalEffectObservationClaimV1,
) -> Result<[u8; 32], ExternalEffectError> {
    Ok(*blake3::hash(&canonical_external_effect_observation_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedExternalEffectObservationV1 {
    claim: ExternalEffectObservationClaimV1,
    policy: ExternalEffectObservationPolicyV1,
}

pub(crate) fn policy_check_external_effect_observation(
    contract: &ExternalEffectContractV1,
    obligation: &ExternalEffectObligationV1,
    profile: &VerifierProfileV1,
    policy: &ExternalEffectObservationPolicyV1,
    claim: ExternalEffectObservationClaimV1,
) -> Result<PolicyCheckedExternalEffectObservationV1, ExternalEffectError> {
    contract.validate()?;
    obligation.validate()?;
    policy.validate_against_profile(profile)?;
    claim.validate()?;
    if contract.obligations.binary_search_by_key(&obligation.id(), ExternalEffectObligationV1::id).is_err() {
        return Err(ExternalEffectError::ObligationOutsideContract);
    }
    if claim.contract_id != contract.id() || claim.obligation_id != obligation.id() {
        return Err(ExternalEffectError::ObservationContextMismatch);
    }
    if claim.verifier_profile_id != profile.id() || claim.verifier_profile_id != policy.verifier_profile_id {
        return Err(ExternalEffectError::VerifierProfileMismatch);
    }
    Ok(PolicyCheckedExternalEffectObservationV1 { claim, policy: policy.clone() })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedExternalEffectObservationV1 {
    checked: PolicyCheckedExternalEffectObservationV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedExternalEffectObservationId,
}

impl AuthenticatedExternalEffectObservationV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedExternalEffectObservationV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, ExternalEffectError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(ExternalEffectError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedExternalEffectObservationId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                checked.claim.id().as_bytes(),
                checked.policy.id().as_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self { checked, authentication_evidence_digest, evidence_id })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedExternalEffectObservationV1 {
    qualified_id: QualifiedExternalEffectObservationId,
    contract_id: ExternalEffectContractId,
    obligation_id: ExternalEffectObligationId,
    policy_id: ExternalEffectObservationPolicyId,
    verifier_profile_id: VerifierProfileId,
    observed_state: ExternalEffectObservedStateV1,
    observed_state_digest: Option<[u8; 32]>,
    resolution_basis_digest: Option<[u8; 32]>,
    qualified_at_unix_ms: u64,
}

pub(crate) fn qualify_external_effect_observation(
    authenticated: &AuthenticatedExternalEffectObservationV1,
    qualified_at_unix_ms: u64,
) -> Result<QualifiedExternalEffectObservationV1, ExternalEffectError> {
    if qualified_at_unix_ms == 0 { return Err(ExternalEffectError::ZeroQualificationTime); }
    let claim = &authenticated.checked.claim;
    let policy = &authenticated.checked.policy;
    check_freshness(
        claim.observed_at_unix_ms,
        qualified_at_unix_ms,
        policy.maximum_observation_age_ms,
        policy.maximum_future_skew_ms,
    )?;
    let qualified_id = QualifiedExternalEffectObservationId(domain_hash_parts(
        QUALIFIED_OBSERVATION_DOMAIN,
        &[
            claim.id().as_bytes(),
            policy.id().as_bytes(),
            authenticated.evidence_id.as_bytes(),
            &qualified_at_unix_ms.to_le_bytes(),
        ],
    ));
    Ok(QualifiedExternalEffectObservationV1 {
        qualified_id,
        contract_id: claim.contract_id,
        obligation_id: claim.obligation_id,
        policy_id: policy.id(),
        verifier_profile_id: claim.verifier_profile_id,
        observed_state: claim.observed_state,
        observed_state_digest: claim.observed_state_digest,
        resolution_basis_digest: claim.resolution_basis_digest,
        qualified_at_unix_ms,
    })
}

impl QualifiedExternalEffectObservationV1 {
    pub fn id(&self) -> QualifiedExternalEffectObservationId { self.qualified_id }
    pub fn obligation_id(&self) -> ExternalEffectObligationId { self.obligation_id }
    pub fn qualified_at_unix_ms(&self) -> u64 { self.qualified_at_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEffectReconciliationRecordV1 {
    schema_version: String,
    contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    observation_ids: Vec<QualifiedExternalEffectObservationId>,
    reconciled_at_unix_ms: u64,
    reconciliation_id: QualifiedExternalEffectReconciliationId,
}

impl ExternalEffectReconciliationRecordV1 {
    pub fn validate(&self) -> Result<(), ExternalEffectError> {
        if self.schema_version != EXTERNAL_EFFECT_RECONCILIATION_RECORD_SCHEMA_V1 {
            return Err(ExternalEffectError::UnsupportedReconciliationSchema(self.schema_version.clone()));
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(ExternalEffectError::ZeroCoverageManifestDigest);
        }
        if self.observation_ids.is_empty() {
            return Err(ExternalEffectError::NoDeclaredExternalEffectObligations);
        }
        if self.observation_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ExternalEffectError::NonCanonicalObservationSet);
        }
        if self.reconciled_at_unix_ms == 0 { return Err(ExternalEffectError::ZeroReconciliationTime); }
        let expected = QualifiedExternalEffectReconciliationId(hash_reconciliation(
            self.contract_id,
            self.coverage_manifest_digest,
            &self.observation_ids,
            self.reconciled_at_unix_ms,
        ));
        if expected != self.reconciliation_id {
            return Err(ExternalEffectError::ReconciliationIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> QualifiedExternalEffectReconciliationId { self.reconciliation_id }
    pub fn contract_id(&self) -> ExternalEffectContractId { self.contract_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn reconciled_at_unix_ms(&self) -> u64 { self.reconciled_at_unix_ms }
}

#[derive(Debug, Clone)]
pub struct QualifiedExternalEffectReconciliationV1 {
    record: ExternalEffectReconciliationRecordV1,
}

impl QualifiedExternalEffectReconciliationV1 {
    pub fn qualify(
        contract: &ExternalEffectContractV1,
        observations: &[QualifiedExternalEffectObservationV1],
        reconciled_at_unix_ms: u64,
    ) -> Result<Self, ExternalEffectError> {
        contract.validate()?;
        if contract.obligations.is_empty() {
            return Err(ExternalEffectError::NoDeclaredExternalEffectObligations);
        }
        if reconciled_at_unix_ms == 0 { return Err(ExternalEffectError::ZeroReconciliationTime); }

        let obligations = contract.obligations.iter().map(|o| (o.id(), o)).collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        let mut observation_ids = Vec::with_capacity(observations.len());
        for observation in observations {
            if observation.contract_id != contract.id() {
                return Err(ExternalEffectError::ObservationContextMismatch);
            }
            if observation.qualified_at_unix_ms != reconciled_at_unix_ms {
                return Err(ExternalEffectError::ObservationEvaluationTimeMismatch);
            }
            if !seen.insert(observation.obligation_id) {
                return Err(ExternalEffectError::DuplicateObservation);
            }
            let obligation = obligations.get(&observation.obligation_id)
                .ok_or(ExternalEffectError::ObservationOutsideContract)?;
            require_predicate_satisfied(obligation.recovery_predicate(), observation)?;
            observation_ids.push(observation.id());
        }
        if seen.len() != obligations.len() {
            return Err(ExternalEffectError::IncompleteObservationSet {
                expected: obligations.len(),
                observed: seen.len(),
            });
        }
        for obligation_id in obligations.keys() {
            if !seen.contains(obligation_id) {
                return Err(ExternalEffectError::MissingObservation { obligation_id: *obligation_id });
            }
        }
        observation_ids.sort();
        let reconciliation_id = QualifiedExternalEffectReconciliationId(hash_reconciliation(
            contract.id(),
            contract.coverage_manifest_digest,
            &observation_ids,
            reconciled_at_unix_ms,
        ));
        let record = ExternalEffectReconciliationRecordV1 {
            schema_version: EXTERNAL_EFFECT_RECONCILIATION_RECORD_SCHEMA_V1.to_owned(),
            contract_id: contract.id(),
            coverage_manifest_digest: contract.coverage_manifest_digest,
            observation_ids,
            reconciled_at_unix_ms,
            reconciliation_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    pub fn rebind(
        record: ExternalEffectReconciliationRecordV1,
        contract: &ExternalEffectContractV1,
        observations: &[QualifiedExternalEffectObservationV1],
    ) -> Result<Self, ExternalEffectError> {
        record.validate()?;
        let fresh = Self::qualify(contract, observations, record.reconciled_at_unix_ms)?;
        if fresh.record != record { return Err(ExternalEffectError::ReconciliationLineageMismatch); }
        Ok(fresh)
    }

    pub fn id(&self) -> QualifiedExternalEffectReconciliationId { self.record.id() }
    pub fn record(&self) -> &ExternalEffectReconciliationRecordV1 { &self.record }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExternalEffectError {
    #[error(transparent)]
    TransitionLineage(#[from] KnownGoodTransitionLineageError),
    #[error(transparent)]
    Verification(#[from] VerificationAdmissionError),
    #[error("unsupported external-effect contract schema: {0}")]
    UnsupportedContractSchema(String),
    #[error("unsupported external-effect observation schema: {0}")]
    UnsupportedObservationSchema(String),
    #[error("unsupported external-effect reconciliation schema: {0}")]
    UnsupportedReconciliationSchema(String),
    #[error("custom external-effect kind is blank, non-canonical, contains controls, or too long")]
    InvalidCustomEffectKind,
    #[error("external-effect destination scope digest must be non-zero")]
    ZeroDestinationScopeDigest,
    #[error("external-effect operation digest must be non-zero")]
    ZeroOperationDigest,
    #[error("external-effect idempotency-key digest, when present, must be non-zero")]
    ZeroIdempotencyKeyDigest,
    #[error("external-effect recovery predicate contains a zero policy/state digest")]
    ZeroRecoveryPredicateDigest,
    #[error("external-effect obligation identity mismatch")]
    ObligationIdentityMismatch,
    #[error("external-effect coverage manifest digest must be non-zero")]
    ZeroCoverageManifestDigest,
    #[error("external-effect source and target realization must differ")]
    SourceEqualsTarget,
    #[error("duplicate external-effect obligation")]
    DuplicateObligation,
    #[error("external-effect obligations are not in strict canonical id order")]
    NonCanonicalObligations,
    #[error("external-effect contract identity mismatch")]
    ContractIdentityMismatch,
    #[error("external-effect obligation is outside the exact contract")]
    ObligationOutsideContract,
    #[error("external-effect observation policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("external-effect maximum observation age must be non-zero")]
    ZeroMaximumObservationAge,
    #[error("external-effect observation verifier profile mismatch")]
    VerifierProfileMismatch,
    #[error("external-effect observation policy identity mismatch")]
    PolicyIdentityMismatch,
    #[error("external-effect observation time must be non-zero")]
    ZeroObservationTime,
    #[error("external-effect raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("known external-effect state requires a non-zero observed-state digest")]
    MissingObservedStateDigest,
    #[error("ABSENT/UNKNOWN external-effect state must not fabricate state/resolution digests")]
    UnexpectedObservationMaterial,
    #[error("COMPENSATED/ACCEPTED_PERSISTENT requires a non-zero resolution basis digest")]
    MissingResolutionBasisDigest,
    #[error("external-effect observation identity mismatch")]
    ObservationIdentityMismatch,
    #[error("external-effect observation belongs to another contract/obligation")]
    ObservationContextMismatch,
    #[error("external-effect authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("external-effect qualification time must be non-zero")]
    ZeroQualificationTime,
    #[error("external-effect evidence is stale: age {age_ms} ms > {allowed_ms} ms")]
    StaleObservation { age_ms: u64, allowed_ms: u64 },
    #[error("external-effect evidence is too far in the future: skew {skew_ms} ms > {allowed_ms} ms")]
    ObservationFromFuture { skew_ms: u64, allowed_ms: u64 },
    #[error("external-effect reconciliation requires at least one declared obligation; no-effects needs a separate proof")]
    NoDeclaredExternalEffectObligations,
    #[error("external-effect qualified observation lies outside the exact contract")]
    ObservationOutsideContract,
    #[error("duplicate qualified observation for one external-effect obligation")]
    DuplicateObservation,
    #[error("external-effect observation qualification time differs from reconciliation evaluation time")]
    ObservationEvaluationTimeMismatch,
    #[error("external-effect observation set incomplete: expected {expected}, observed {observed}")]
    IncompleteObservationSet { expected: usize, observed: usize },
    #[error("missing qualified observation for external-effect obligation {obligation_id:?}")]
    MissingObservation { obligation_id: ExternalEffectObligationId },
    #[error("external-effect observation does not satisfy the obligation recovery predicate")]
    RecoveryPredicateNotSatisfied,
    #[error("external-effect resolution basis does not match the exact compensation/acceptance policy")]
    ResolutionBasisMismatch,
    #[error("external-effect source-equivalent state does not match the exact source-state digest")]
    SourceStateDigestMismatch,
    #[error("external-effect observation ids are not in strict canonical order")]
    NonCanonicalObservationSet,
    #[error("external-effect reconciliation time must be non-zero")]
    ZeroReconciliationTime,
    #[error("external-effect reconciliation identity mismatch")]
    ReconciliationIdentityMismatch,
    #[error("persisted external-effect reconciliation does not match exact live proof set")]
    ReconciliationLineageMismatch,
}

fn require_predicate_satisfied(
    predicate: &ExternalEffectRecoveryPredicateV1,
    observation: &QualifiedExternalEffectObservationV1,
) -> Result<(), ExternalEffectError> {
    match predicate {
        ExternalEffectRecoveryPredicateV1::MustBeAbsent => {
            if observation.observed_state != ExternalEffectObservedStateV1::Absent {
                return Err(ExternalEffectError::RecoveryPredicateNotSatisfied);
            }
        }
        ExternalEffectRecoveryPredicateV1::MustMatchSourceState { source_state_digest } => {
            if observation.observed_state != ExternalEffectObservedStateV1::SourceEquivalent {
                return Err(ExternalEffectError::RecoveryPredicateNotSatisfied);
            }
            if observation.observed_state_digest != Some(*source_state_digest) {
                return Err(ExternalEffectError::SourceStateDigestMismatch);
            }
        }
        ExternalEffectRecoveryPredicateV1::MustBeCompensated { compensation_contract_digest } => {
            if observation.observed_state != ExternalEffectObservedStateV1::Compensated {
                return Err(ExternalEffectError::RecoveryPredicateNotSatisfied);
            }
            if observation.resolution_basis_digest != Some(*compensation_contract_digest) {
                return Err(ExternalEffectError::ResolutionBasisMismatch);
            }
        }
        ExternalEffectRecoveryPredicateV1::MayPersistUnderPolicy { acceptance_policy_digest } => {
            if observation.observed_state != ExternalEffectObservedStateV1::AcceptedPersistent {
                return Err(ExternalEffectError::RecoveryPredicateNotSatisfied);
            }
            if observation.resolution_basis_digest != Some(*acceptance_policy_digest) {
                return Err(ExternalEffectError::ResolutionBasisMismatch);
            }
        }
    }
    Ok(())
}

fn validate_class(class: &ExternalEffectClassV1) -> Result<(), ExternalEffectError> {
    if let ExternalEffectClassV1::Custom { kind_id } = class {
        let trimmed = kind_id.trim();
        if trimmed.is_empty()
            || trimmed != kind_id
            || trimmed.len() > MAX_CUSTOM_KIND_BYTES
            || trimmed.chars().any(char::is_control)
        {
            return Err(ExternalEffectError::InvalidCustomEffectKind);
        }
    }
    Ok(())
}

fn validate_obligation_material(
    destination_scope_digest: [u8; 32],
    operation_digest: [u8; 32],
    idempotency_key_digest: Option<[u8; 32]>,
    predicate: &ExternalEffectRecoveryPredicateV1,
) -> Result<(), ExternalEffectError> {
    if destination_scope_digest == [0; 32] { return Err(ExternalEffectError::ZeroDestinationScopeDigest); }
    if operation_digest == [0; 32] { return Err(ExternalEffectError::ZeroOperationDigest); }
    if idempotency_key_digest == Some([0; 32]) { return Err(ExternalEffectError::ZeroIdempotencyKeyDigest); }
    let predicate_digest = match predicate {
        ExternalEffectRecoveryPredicateV1::MustBeAbsent => None,
        ExternalEffectRecoveryPredicateV1::MustMatchSourceState { source_state_digest } => Some(*source_state_digest),
        ExternalEffectRecoveryPredicateV1::MustBeCompensated { compensation_contract_digest } => Some(*compensation_contract_digest),
        ExternalEffectRecoveryPredicateV1::MayPersistUnderPolicy { acceptance_policy_digest } => Some(*acceptance_policy_digest),
    };
    if predicate_digest == Some([0; 32]) { return Err(ExternalEffectError::ZeroRecoveryPredicateDigest); }
    Ok(())
}

fn validate_observation_material(
    observed_at_unix_ms: u64,
    state: ExternalEffectObservedStateV1,
    observed_state_digest: Option<[u8; 32]>,
    resolution_basis_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> Result<(), ExternalEffectError> {
    if observed_at_unix_ms == 0 { return Err(ExternalEffectError::ZeroObservationTime); }
    if raw_evidence_digest == [0; 32] { return Err(ExternalEffectError::ZeroRawEvidenceDigest); }
    match state {
        ExternalEffectObservedStateV1::Absent | ExternalEffectObservedStateV1::Unknown => {
            if observed_state_digest.is_some() || resolution_basis_digest.is_some() {
                return Err(ExternalEffectError::UnexpectedObservationMaterial);
            }
        }
        ExternalEffectObservedStateV1::SourceEquivalent
        | ExternalEffectObservedStateV1::PresentUnreconciled => {
            if observed_state_digest.is_none() || observed_state_digest == Some([0; 32]) {
                return Err(ExternalEffectError::MissingObservedStateDigest);
            }
            if resolution_basis_digest.is_some() {
                return Err(ExternalEffectError::UnexpectedObservationMaterial);
            }
        }
        ExternalEffectObservedStateV1::Compensated
        | ExternalEffectObservedStateV1::AcceptedPersistent => {
            if observed_state_digest.is_none() || observed_state_digest == Some([0; 32]) {
                return Err(ExternalEffectError::MissingObservedStateDigest);
            }
            if resolution_basis_digest.is_none() || resolution_basis_digest == Some([0; 32]) {
                return Err(ExternalEffectError::MissingResolutionBasisDigest);
            }
        }
    }
    Ok(())
}

fn check_freshness(
    observed_at_unix_ms: u64,
    qualified_at_unix_ms: u64,
    maximum_age_ms: u64,
    maximum_future_skew_ms: u64,
) -> Result<(), ExternalEffectError> {
    if observed_at_unix_ms > qualified_at_unix_ms {
        let skew = observed_at_unix_ms - qualified_at_unix_ms;
        if skew > maximum_future_skew_ms {
            return Err(ExternalEffectError::ObservationFromFuture { skew_ms: skew, allowed_ms: maximum_future_skew_ms });
        }
        return Ok(());
    }
    let age = qualified_at_unix_ms - observed_at_unix_ms;
    if age > maximum_age_ms {
        return Err(ExternalEffectError::StaleObservation { age_ms: age, allowed_ms: maximum_age_ms });
    }
    Ok(())
}

fn hash_obligation(
    class: &ExternalEffectClassV1,
    destination_scope_digest: [u8; 32],
    operation_digest: [u8; 32],
    idempotency_key_digest: Option<[u8; 32]>,
    predicate: &ExternalEffectRecoveryPredicateV1,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(OBLIGATION_DOMAIN);
    encode_class(&mut hasher, class);
    hasher.update(&destination_scope_digest);
    hasher.update(&operation_digest);
    hash_optional_digest(&mut hasher, idempotency_key_digest);
    encode_predicate(&mut hasher, predicate);
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn hash_contract(
    intent_id: KnownGoodExecutionIntentId,
    lineage_id: KnownGoodTransitionLineageId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_id: TargetRealizationId,
    target_id: TargetRealizationId,
    coverage_manifest_digest: [u8; 32],
    obligations: &[ExternalEffectObligationV1],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTRACT_DOMAIN);
    hasher.update(intent_id.as_bytes());
    hasher.update(lineage_id.as_bytes());
    hasher.update(attempt_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(source_id.as_bytes());
    hasher.update(target_id.as_bytes());
    hasher.update(&coverage_manifest_digest);
    hasher.update(&(obligations.len() as u64).to_le_bytes());
    for obligation in obligations { hasher.update(obligation.id().as_bytes()); }
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn hash_observation_claim(
    contract_id: ExternalEffectContractId,
    obligation_id: ExternalEffectObligationId,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    state: ExternalEffectObservedStateV1,
    observed_state_digest: Option<[u8; 32]>,
    resolution_basis_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLAIM_DOMAIN);
    hasher.update(contract_id.as_bytes());
    hasher.update(obligation_id.as_bytes());
    hasher.update(verifier_profile_id.as_bytes());
    hasher.update(&observed_at_unix_ms.to_le_bytes());
    hasher.update(&[state.tag()]);
    hash_optional_digest(&mut hasher, observed_state_digest);
    hash_optional_digest(&mut hasher, resolution_basis_digest);
    hasher.update(&raw_evidence_digest);
    *hasher.finalize().as_bytes()
}

fn hash_reconciliation(
    contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    observations: &[QualifiedExternalEffectObservationId],
    reconciled_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECONCILIATION_DOMAIN);
    hasher.update(contract_id.as_bytes());
    hasher.update(&coverage_manifest_digest);
    hasher.update(&(observations.len() as u64).to_le_bytes());
    for id in observations { hasher.update(id.as_bytes()); }
    hasher.update(&reconciled_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn encode_class(hasher: &mut blake3::Hasher, class: &ExternalEffectClassV1) {
    match class {
        ExternalEffectClassV1::DatabaseMutation => hasher.update(&[1]),
        ExternalEffectClassV1::MessageEmission => hasher.update(&[2]),
        ExternalEffectClassV1::ExternalApiMutation => hasher.update(&[3]),
        ExternalEffectClassV1::NetworkControlPlaneMutation => hasher.update(&[4]),
        ExternalEffectClassV1::StorageMutation => hasher.update(&[5]),
        ExternalEffectClassV1::IdentityOrCredentialMutation => hasher.update(&[6]),
        ExternalEffectClassV1::DeviceActuation => hasher.update(&[7]),
        ExternalEffectClassV1::Custom { kind_id } => {
            hasher.update(&[255]);
            hasher.update(&(kind_id.len() as u64).to_le_bytes());
            hasher.update(kind_id.as_bytes());
        }
    }
}

fn encode_predicate(hasher: &mut blake3::Hasher, predicate: &ExternalEffectRecoveryPredicateV1) {
    match predicate {
        ExternalEffectRecoveryPredicateV1::MustBeAbsent => hasher.update(&[1]),
        ExternalEffectRecoveryPredicateV1::MustMatchSourceState { source_state_digest } => {
            hasher.update(&[2]); hasher.update(source_state_digest);
        }
        ExternalEffectRecoveryPredicateV1::MustBeCompensated { compensation_contract_digest } => {
            hasher.update(&[3]); hasher.update(compensation_contract_digest);
        }
        ExternalEffectRecoveryPredicateV1::MayPersistUnderPolicy { acceptance_policy_digest } => {
            hasher.update(&[4]); hasher.update(acceptance_policy_digest);
        }
    }
}

fn encode_optional_digest(out: &mut Vec<u8>, digest: Option<[u8; 32]>) {
    match digest {
        Some(digest) => { out.push(1); out.extend_from_slice(&digest); }
        None => out.push(0),
    }
}

fn hash_optional_digest(hasher: &mut blake3::Hasher, digest: Option<[u8; 32]>) {
    match digest {
        Some(digest) => { hasher.update(&[1]); hasher.update(&digest); }
        None => { hasher.update(&[0]); }
    }
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts { hasher.update(part); }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_control_plane_is_first_class_external_effect() {
        assert_ne!(
            ExternalEffectClassV1::NetworkControlPlaneMutation,
            ExternalEffectClassV1::StorageMutation
        );
    }

    #[test]
    fn unknown_effect_cannot_fabricate_resolution_material() {
        assert_eq!(
            validate_observation_material(
                1,
                ExternalEffectObservedStateV1::Unknown,
                Some([1; 32]),
                None,
                [2; 32],
            ).unwrap_err(),
            ExternalEffectError::UnexpectedObservationMaterial
        );
    }

    #[test]
    fn observation_wire_is_domain_separated() {
        assert_ne!(WIRE_DOMAIN, CLAIM_DOMAIN);
        assert_eq!(
            EXTERNAL_EFFECT_OBSERVATION_AUTH_PURPOSE,
            "symthaea.continuity.external-effect-observation.v1"
        );
    }
}
