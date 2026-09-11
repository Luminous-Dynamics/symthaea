// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Attempt-specific, authorized external-effect scope at the physical execution gate.
//!
//! The effect plan is authorized before capability minting. This layer derives the
//! exact attempt-specific contract from that plan, binds it into a durable envelope,
//! and requires a protected commitment to that envelope tied to the exact qualified
//! post-intent execution-journal anchor before the physical adapter receives a token.
//!
//! `AuthorizedPlan != AttemptContract != DurableScope != ReadyForPhysicalExecution`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionBackendProfileV1, ExecutionEpochAnchorModeV1,
};
use crate::execution_coordinator::{
    KnownGoodExecutionCoordinatorError, PendingAnchoredKnownGoodExecutionV1,
    ReadyKnownGoodExecutionAttemptV1, prepare_known_good_execution,
};
use crate::execution_journal::ReconstructedExecutionJournalV1;
use crate::execution_journal_anchor::{
    ExecutionJournalAnchorError, ExecutionJournalAnchorProfileId,
    ExecutionJournalAnchorProfileV1, QualifiedExecutionJournalAnchorId,
    QualifiedExecutionJournalAnchorV1,
};
use crate::external_effect_authority::{
    EffectScopedKnownGoodBoundEligibilityV1, ExternalEffectAuthorityError,
    ExternalEffectPlanId, ExternalEffectPlanV1, QualifiedExternalEffectAuthorizationId,
    QualifiedExternalEffectAuthorizationV1,
};
use crate::external_effects::{
    ExternalEffectContractId, ExternalEffectContractV1, ExternalEffectError,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{KnownGoodExecutionIntentId, KnownGoodTransitionLineageId};
use crate::trusted_commit_epoch::{
    QualifiedTrustedCommitEpochId, QualifiedTrustedCommitEpochV1,
};
use crate::witness::TargetRealizationId;

pub const EFFECT_SCOPED_EXECUTION_ENVELOPE_SCHEMA_V1: &str =
    "symthaea-continuity-effect-scoped-execution-envelope-v1";
pub const EFFECT_SCOPE_COMMITMENT_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-effect-scope-commitment-claim-v1";
pub const EFFECT_SCOPE_COMMITMENT_AUTH_PURPOSE: &str =
    "symthaea.continuity.effect-scope-commitment.v1";

const ENVELOPE_DOMAIN: &[u8] = b"symthaea.continuity.effect-scoped-execution-envelope.v1\0";
const COMMITMENT_CLAIM_DOMAIN: &[u8] =
    b"symthaea.continuity.effect-scope-commitment-claim.v1\0";
const COMMITMENT_WIRE_DOMAIN: &[u8] =
    b"symthaea.continuity.effect-scope-commitment-wire.v1\0";
const COMMITMENT_AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-effect-scope-commitment.v1\0";
const COMMITMENT_QUALIFIED_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-effect-scope-commitment.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(EffectScopedExecutionEnvelopeId);
digest_id!(EffectScopeCommitmentClaimId);
digest_id!(AuthenticatedEffectScopeCommitmentId);
digest_id!(QualifiedEffectScopeCommitmentId);

/// Durable semantic envelope that binds the exact physical attempt to the exact
/// pre-authorized effect plan and its derived attempt-specific contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectScopedExecutionEnvelopeV1 {
    schema_version: String,
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    transition_lineage_id: KnownGoodTransitionLineageId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    effect_plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    effect_contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    envelope_id: EffectScopedExecutionEnvelopeId,
}

impl EffectScopedExecutionEnvelopeV1 {
    fn new(
        pending: &PendingAnchoredKnownGoodExecutionV1,
        plan: &ExternalEffectPlanV1,
        authorization: &QualifiedExternalEffectAuthorizationV1,
        contract: &ExternalEffectContractV1,
    ) -> Result<Self, EffectScopedExecutionError> {
        pending.intent().validate()?;
        plan.validate()?;
        contract.validate()?;
        let lineage = pending.intent().lineage();
        if contract.attempt_id() != pending.attempt_id()
            || contract.subject_id() != lineage.subject_id()
            || contract.coverage_manifest_digest() != plan.coverage_manifest_digest()
            || contract.obligations() != plan.obligations()
            || authorization.plan_id() != plan.id()
            || authorization.subject_id() != lineage.subject_id()
            || authorization.target_realization_id() != lineage.target_realization_id()
            || authorization.distributed_context_id() != lineage.distributed_context_id()
            || authorization.coverage_manifest_digest() != plan.coverage_manifest_digest()
        {
            return Err(EffectScopedExecutionError::AuthorizedPlanContractMismatch);
        }

        let envelope_id = EffectScopedExecutionEnvelopeId(hash_envelope(
            pending.intent().id(),
            pending.attempt_id(),
            lineage.id(),
            lineage.subject_id(),
            lineage.source_realization_id(),
            lineage.target_realization_id(),
            lineage.distributed_context_id(),
            plan.id(),
            authorization.id(),
            contract.id(),
            plan.coverage_manifest_digest(),
        ));
        Ok(Self {
            schema_version: EFFECT_SCOPED_EXECUTION_ENVELOPE_SCHEMA_V1.to_owned(),
            known_good_intent_id: pending.intent().id(),
            attempt_id: pending.attempt_id(),
            transition_lineage_id: lineage.id(),
            subject_id: lineage.subject_id(),
            source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(),
            distributed_context_id: lineage.distributed_context_id(),
            effect_plan_id: plan.id(),
            effect_authorization_id: authorization.id(),
            effect_contract_id: contract.id(),
            coverage_manifest_digest: plan.coverage_manifest_digest(),
            envelope_id,
        })
    }

    pub fn validate(&self) -> Result<(), EffectScopedExecutionError> {
        if self.schema_version != EFFECT_SCOPED_EXECUTION_ENVELOPE_SCHEMA_V1 {
            return Err(EffectScopedExecutionError::UnsupportedEnvelopeSchema(
                self.schema_version.clone(),
            ));
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(EffectScopedExecutionError::ZeroCoverageManifestDigest);
        }
        if self.source_realization_id == self.target_realization_id {
            return Err(EffectScopedExecutionError::SourceEqualsTarget);
        }
        let expected = EffectScopedExecutionEnvelopeId(hash_envelope(
            self.known_good_intent_id,
            self.attempt_id,
            self.transition_lineage_id,
            self.subject_id,
            self.source_realization_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.effect_plan_id,
            self.effect_authorization_id,
            self.effect_contract_id,
            self.coverage_manifest_digest,
        ));
        if expected != self.envelope_id {
            return Err(EffectScopedExecutionError::EnvelopeIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> EffectScopedExecutionEnvelopeId { self.envelope_id }
    pub fn known_good_intent_id(&self) -> KnownGoodExecutionIntentId { self.known_good_intent_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn effect_plan_id(&self) -> ExternalEffectPlanId { self.effect_plan_id }
    pub fn effect_authorization_id(&self) -> QualifiedExternalEffectAuthorizationId {
        self.effect_authorization_id
    }
    pub fn effect_contract_id(&self) -> ExternalEffectContractId { self.effect_contract_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
}

/// Protected platform claim that the exact effect-scoped envelope is committed
/// alongside the exact already-qualified post-intent journal anchor. The journal
/// anchor supplies monotonic/rollback-resistant sequence semantics; this claim adds
/// the authorized effect scope to that exact protected world without changing V1
/// execution-journal identities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectScopeCommitmentClaimV1 {
    schema_version: String,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    effect_plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    effect_contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    committed_at_unix_ms: u64,
    raw_scope_evidence_digest: [u8; 32],
    claim_id: EffectScopeCommitmentClaimId,
}

impl EffectScopeCommitmentClaimV1 {
    pub fn new(
        profile: &ExecutionJournalAnchorProfileV1,
        envelope: &EffectScopedExecutionEnvelopeV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        raw_scope_evidence_digest: [u8; 32],
    ) -> Result<Self, EffectScopedExecutionError> {
        profile.validate()?;
        envelope.validate()?;
        if raw_scope_evidence_digest == [0; 32] {
            return Err(EffectScopedExecutionError::ZeroScopeEvidenceDigest);
        }
        if profile.id() != journal_anchor.profile_id()
            || profile.root_epoch() != journal_anchor.root_epoch()
            || envelope.subject_id() != journal_anchor.subject_id()
        {
            return Err(EffectScopedExecutionError::ScopeAnchorRootMismatch);
        }
        let committed_at_unix_ms = journal_anchor.anchored_at_unix_ms();
        let claim_id = EffectScopeCommitmentClaimId(hash_commitment_claim(
            profile.id(),
            profile.root_epoch(),
            journal_anchor.id(),
            journal_anchor.trusted_epoch_id(),
            envelope.id(),
            envelope.known_good_intent_id(),
            envelope.attempt_id(),
            envelope.subject_id(),
            envelope.effect_plan_id(),
            envelope.effect_authorization_id(),
            envelope.effect_contract_id(),
            envelope.coverage_manifest_digest(),
            committed_at_unix_ms,
            raw_scope_evidence_digest,
        ));
        Ok(Self {
            schema_version: EFFECT_SCOPE_COMMITMENT_CLAIM_SCHEMA_V1.to_owned(),
            profile_id: profile.id(),
            root_epoch: profile.root_epoch(),
            journal_anchor_id: journal_anchor.id(),
            trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            envelope_id: envelope.id(),
            known_good_intent_id: envelope.known_good_intent_id(),
            attempt_id: envelope.attempt_id(),
            subject_id: envelope.subject_id(),
            effect_plan_id: envelope.effect_plan_id(),
            effect_authorization_id: envelope.effect_authorization_id(),
            effect_contract_id: envelope.effect_contract_id(),
            coverage_manifest_digest: envelope.coverage_manifest_digest(),
            committed_at_unix_ms,
            raw_scope_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), EffectScopedExecutionError> {
        if self.schema_version != EFFECT_SCOPE_COMMITMENT_CLAIM_SCHEMA_V1 {
            return Err(EffectScopedExecutionError::UnsupportedScopeClaimSchema(
                self.schema_version.clone(),
            ));
        }
        if self.root_epoch == 0 {
            return Err(EffectScopedExecutionError::ZeroScopeRootEpoch);
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(EffectScopedExecutionError::ZeroCoverageManifestDigest);
        }
        if self.committed_at_unix_ms == 0 {
            return Err(EffectScopedExecutionError::ZeroScopeCommitmentTime);
        }
        if self.raw_scope_evidence_digest == [0; 32] {
            return Err(EffectScopedExecutionError::ZeroScopeEvidenceDigest);
        }
        let expected = EffectScopeCommitmentClaimId(hash_commitment_claim(
            self.profile_id,
            self.root_epoch,
            self.journal_anchor_id,
            self.trusted_epoch_id,
            self.envelope_id,
            self.known_good_intent_id,
            self.attempt_id,
            self.subject_id,
            self.effect_plan_id,
            self.effect_authorization_id,
            self.effect_contract_id,
            self.coverage_manifest_digest,
            self.committed_at_unix_ms,
            self.raw_scope_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(EffectScopedExecutionError::ScopeClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> EffectScopeCommitmentClaimId { self.claim_id }
}

pub fn canonical_effect_scope_commitment_claim_bytes(
    claim: &EffectScopeCommitmentClaimV1,
) -> Result<Vec<u8>, EffectScopedExecutionError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(640);
    out.extend_from_slice(COMMITMENT_WIRE_DOMAIN);
    out.extend_from_slice(claim.profile_id.as_bytes());
    out.extend_from_slice(&claim.root_epoch.to_le_bytes());
    out.extend_from_slice(claim.journal_anchor_id.as_bytes());
    out.extend_from_slice(claim.trusted_epoch_id.as_bytes());
    out.extend_from_slice(claim.envelope_id.as_bytes());
    out.extend_from_slice(claim.known_good_intent_id.as_bytes());
    out.extend_from_slice(claim.attempt_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.effect_plan_id.as_bytes());
    out.extend_from_slice(claim.effect_authorization_id.as_bytes());
    out.extend_from_slice(claim.effect_contract_id.as_bytes());
    out.extend_from_slice(&claim.coverage_manifest_digest);
    out.extend_from_slice(&claim.committed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_scope_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_effect_scope_commitment_claim_digest(
    claim: &EffectScopeCommitmentClaimV1,
) -> Result<[u8; 32], EffectScopedExecutionError> {
    Ok(*blake3::hash(&canonical_effect_scope_commitment_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedEffectScopeCommitmentV1 {
    claim: EffectScopeCommitmentClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedEffectScopeCommitmentId,
}

impl AuthenticatedEffectScopeCommitmentV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: EffectScopeCommitmentClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, EffectScopedExecutionError> {
        claim.validate()?;
        profile.validate()?;
        if claim.profile_id != profile.id() || claim.root_epoch != profile.root_epoch() {
            return Err(EffectScopedExecutionError::ScopeAnchorRootMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(EffectScopedExecutionError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedEffectScopeCommitmentId(domain_hash_parts(
            COMMITMENT_AUTH_DOMAIN,
            &[
                claim.id().as_bytes(),
                profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self {
            claim,
            profile,
            authentication_evidence_digest,
            evidence_id,
        })
    }
}

/// Non-Serde proof that the exact effect-scoped envelope is protected by the same
/// exact root and exact post-intent journal anchor required for physical release.
#[derive(Debug, Clone)]
pub struct QualifiedEffectScopeCommitmentV1 {
    commitment_id: QualifiedEffectScopeCommitmentId,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    effect_plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    effect_contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    committed_at_unix_ms: u64,
    authenticated_evidence_id: AuthenticatedEffectScopeCommitmentId,
}

impl QualifiedEffectScopeCommitmentV1 {
    pub(crate) fn qualify(
        envelope: &EffectScopedExecutionEnvelopeV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        authenticated: &AuthenticatedEffectScopeCommitmentV1,
    ) -> Result<Self, EffectScopedExecutionError> {
        envelope.validate()?;
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        let claim = &authenticated.claim;
        if authenticated.profile.id() != journal_anchor.profile_id()
            || authenticated.profile.root_epoch() != journal_anchor.root_epoch()
            || claim.profile_id != journal_anchor.profile_id()
            || claim.root_epoch != journal_anchor.root_epoch()
            || claim.journal_anchor_id != journal_anchor.id()
            || claim.trusted_epoch_id != journal_anchor.trusted_epoch_id()
            || claim.committed_at_unix_ms != journal_anchor.anchored_at_unix_ms()
        {
            return Err(EffectScopedExecutionError::ScopeAnchorContextMismatch);
        }
        if claim.envelope_id != envelope.id()
            || claim.known_good_intent_id != envelope.known_good_intent_id()
            || claim.attempt_id != envelope.attempt_id()
            || claim.subject_id != envelope.subject_id()
            || claim.effect_plan_id != envelope.effect_plan_id()
            || claim.effect_authorization_id != envelope.effect_authorization_id()
            || claim.effect_contract_id != envelope.effect_contract_id()
            || claim.coverage_manifest_digest != envelope.coverage_manifest_digest()
        {
            return Err(EffectScopedExecutionError::ScopeEnvelopeMismatch);
        }
        let commitment_id = QualifiedEffectScopeCommitmentId(domain_hash_parts(
            COMMITMENT_QUALIFIED_DOMAIN,
            &[
                envelope.id().as_bytes(),
                journal_anchor.id().as_bytes(),
                claim.id().as_bytes(),
                authenticated.profile.id().as_bytes(),
                &authenticated.profile.root_epoch().to_le_bytes(),
                authenticated.evidence_id.as_bytes(),
            ],
        ));
        Ok(Self {
            commitment_id,
            profile_id: authenticated.profile.id(),
            root_epoch: authenticated.profile.root_epoch(),
            journal_anchor_id: journal_anchor.id(),
            trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            envelope_id: envelope.id(),
            attempt_id: envelope.attempt_id(),
            subject_id: envelope.subject_id(),
            effect_plan_id: envelope.effect_plan_id(),
            effect_authorization_id: envelope.effect_authorization_id(),
            effect_contract_id: envelope.effect_contract_id(),
            coverage_manifest_digest: envelope.coverage_manifest_digest(),
            committed_at_unix_ms: journal_anchor.anchored_at_unix_ms(),
            authenticated_evidence_id: authenticated.evidence_id,
        })
    }

    pub fn id(&self) -> QualifiedEffectScopeCommitmentId { self.commitment_id }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn envelope_id(&self) -> EffectScopedExecutionEnvelopeId { self.envelope_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn effect_plan_id(&self) -> ExternalEffectPlanId { self.effect_plan_id }
    pub fn effect_authorization_id(&self) -> QualifiedExternalEffectAuthorizationId {
        self.effect_authorization_id
    }
    pub fn effect_contract_id(&self) -> ExternalEffectContractId { self.effect_contract_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn committed_at_unix_ms(&self) -> u64 { self.committed_at_unix_ms }
}

/// Non-Clone pending state. The existing physical attempt remains hidden inside the
/// V1 coordinator until both journal durability and exact protected effect scope are
/// proven.
#[derive(Debug)]
pub struct PendingEffectScopedExecutionV1 {
    pending: PendingAnchoredKnownGoodExecutionV1,
    plan: ExternalEffectPlanV1,
    authorization: QualifiedExternalEffectAuthorizationV1,
    contract: ExternalEffectContractV1,
    envelope: EffectScopedExecutionEnvelopeV1,
}

impl PendingEffectScopedExecutionV1 {
    pub fn intent(&self) -> &crate::transition_lineage::KnownGoodBoundExecutionAttemptIntentV1 {
        self.pending.intent()
    }
    pub fn plan(&self) -> &ExternalEffectPlanV1 { &self.plan }
    pub fn authorization(&self) -> &QualifiedExternalEffectAuthorizationV1 { &self.authorization }
    pub fn contract(&self) -> &ExternalEffectContractV1 { &self.contract }
    pub fn envelope(&self) -> &EffectScopedExecutionEnvelopeV1 { &self.envelope }

    pub fn release_after_durable_scope(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        scope_commitment: &QualifiedEffectScopeCommitmentV1,
    ) -> Result<ReadyEffectScopedExecutionV1, EffectScopedExecutionError> {
        self.envelope.validate()?;
        self.contract.validate()?;
        self.plan.validate()?;
        if scope_commitment.journal_anchor_id() != journal_anchor.id()
            || scope_commitment.envelope_id() != self.envelope.id()
            || scope_commitment.attempt_id() != self.envelope.attempt_id()
            || scope_commitment.subject_id() != self.envelope.subject_id()
            || scope_commitment.effect_plan_id() != self.plan.id()
            || scope_commitment.effect_authorization_id() != self.authorization.id()
            || scope_commitment.effect_contract_id() != self.contract.id()
            || scope_commitment.coverage_manifest_digest() != self.plan.coverage_manifest_digest()
        {
            return Err(EffectScopedExecutionError::ScopeReleaseMismatch);
        }
        let ready = self.pending.release_after_durable_anchor(journal, journal_anchor)?;
        if ready.attempt_id() != self.envelope.attempt_id()
            || ready.known_good_intent_id() != self.envelope.known_good_intent_id()
            || ready.subject_id() != self.envelope.subject_id()
            || ready.source_realization_id() != self.envelope.source_realization_id()
            || ready.target_realization_id() != self.envelope.target_realization_id()
        {
            return Err(EffectScopedExecutionError::ReadyAttemptEnvelopeMismatch);
        }
        Ok(ReadyEffectScopedExecutionV1 {
            ready,
            contract: self.contract,
            envelope: self.envelope,
            scope_commitment_id: scope_commitment.id(),
        })
    }
}

/// The only physical token in the effect-scoped path. It retains the exact contract
/// and protected scope identity through backend execution.
#[derive(Debug)]
pub struct ReadyEffectScopedExecutionV1 {
    ready: ReadyKnownGoodExecutionAttemptV1,
    contract: ExternalEffectContractV1,
    envelope: EffectScopedExecutionEnvelopeV1,
    scope_commitment_id: QualifiedEffectScopeCommitmentId,
}

impl ReadyEffectScopedExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.ready.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.ready.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.ready.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.ready.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.ready.target_realization_id() }
    pub fn effect_contract(&self) -> &ExternalEffectContractV1 { &self.contract }
    pub fn envelope(&self) -> &EffectScopedExecutionEnvelopeV1 { &self.envelope }
    pub fn scope_commitment_id(&self) -> QualifiedEffectScopeCommitmentId {
        self.scope_commitment_id
    }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, EffectScopedExecutionError> {
        Ok(self.ready.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_effect_scoped_execution(
    scoped: EffectScopedKnownGoodBoundEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    backend: &ExecutionBackendProfileV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingEffectScopedExecutionV1, EffectScopedExecutionError> {
    let (bound, plan, authorization) = scoped.into_parts();
    let pending = prepare_known_good_execution(
        bound,
        current_epoch,
        previous_epoch,
        epoch_anchor_mode,
        predecessor_journal_anchor,
        backend,
        session_generation,
        session_nonce,
    )?;
    let contract = ExternalEffectContractV1::new(
        pending.intent(),
        plan.coverage_manifest_digest(),
        plan.obligations().to_vec(),
    )?;
    let envelope = EffectScopedExecutionEnvelopeV1::new(
        &pending,
        &plan,
        &authorization,
        &contract,
    )?;
    Ok(PendingEffectScopedExecutionV1 {
        pending,
        plan,
        authorization,
        contract,
        envelope,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EffectScopedExecutionError {
    #[error(transparent)]
    Coordinator(#[from] KnownGoodExecutionCoordinatorError),
    #[error(transparent)]
    EffectAuthority(#[from] ExternalEffectAuthorityError),
    #[error(transparent)]
    ExternalEffect(#[from] ExternalEffectError),
    #[error(transparent)]
    JournalAnchor(#[from] ExecutionJournalAnchorError),
    #[error("unsupported effect-scoped execution envelope schema: {0}")]
    UnsupportedEnvelopeSchema(String),
    #[error("unsupported effect-scope commitment claim schema: {0}")]
    UnsupportedScopeClaimSchema(String),
    #[error("effect-scoped execution coverage manifest digest must be non-zero")]
    ZeroCoverageManifestDigest,
    #[error("effect-scoped execution source realization equals target realization")]
    SourceEqualsTarget,
    #[error("effect-scoped execution envelope identity mismatch")]
    EnvelopeIdentityMismatch,
    #[error("attempt-specific effect contract does not exactly derive from authorized plan")]
    AuthorizedPlanContractMismatch,
    #[error("effect-scope commitment root epoch must be non-zero")]
    ZeroScopeRootEpoch,
    #[error("effect-scope commitment time must be non-zero")]
    ZeroScopeCommitmentTime,
    #[error("effect-scope raw protected evidence digest must be non-zero")]
    ZeroScopeEvidenceDigest,
    #[error("effect-scope commitment root/profile differs from exact journal anchor")]
    ScopeAnchorRootMismatch,
    #[error("effect-scope commitment claim identity mismatch")]
    ScopeClaimIdentityMismatch,
    #[error("effect-scope commitment authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("authenticated effect-scope commitment does not bind exact journal anchor context")]
    ScopeAnchorContextMismatch,
    #[error("authenticated effect-scope commitment does not bind exact durable envelope")]
    ScopeEnvelopeMismatch,
    #[error("physical release supplied a different effect scope commitment")]
    ScopeReleaseMismatch,
    #[error("ready physical attempt differs from exact effect-scoped envelope")]
    ReadyAttemptEnvelopeMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_envelope(
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    lineage_id: KnownGoodTransitionLineageId,
    subject_id: ContinuitySubjectId,
    source_id: TargetRealizationId,
    target_id: TargetRealizationId,
    context_id: DistributedStateContextId,
    plan_id: ExternalEffectPlanId,
    authorization_id: QualifiedExternalEffectAuthorizationId,
    contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
) -> [u8; 32] {
    domain_hash_parts(
        ENVELOPE_DOMAIN,
        &[
            known_good_intent_id.as_bytes(),
            attempt_id.as_bytes(),
            lineage_id.as_bytes(),
            subject_id.as_bytes(),
            source_id.as_bytes(),
            target_id.as_bytes(),
            context_id.as_bytes(),
            plan_id.as_bytes(),
            authorization_id.as_bytes(),
            contract_id.as_bytes(),
            &coverage_manifest_digest,
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn hash_commitment_claim(
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    plan_id: ExternalEffectPlanId,
    authorization_id: QualifiedExternalEffectAuthorizationId,
    contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    committed_at_unix_ms: u64,
    raw_scope_evidence_digest: [u8; 32],
) -> [u8; 32] {
    domain_hash_parts(
        COMMITMENT_CLAIM_DOMAIN,
        &[
            profile_id.as_bytes(),
            &root_epoch.to_le_bytes(),
            journal_anchor_id.as_bytes(),
            trusted_epoch_id.as_bytes(),
            envelope_id.as_bytes(),
            known_good_intent_id.as_bytes(),
            attempt_id.as_bytes(),
            subject_id.as_bytes(),
            plan_id.as_bytes(),
            authorization_id.as_bytes(),
            contract_id.as_bytes(),
            &coverage_manifest_digest,
            &committed_at_unix_ms.to_le_bytes(),
            &raw_scope_evidence_digest,
        ],
    )
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_and_commitment_domains_are_distinct() {
        assert_ne!(ENVELOPE_DOMAIN, COMMITMENT_CLAIM_DOMAIN);
    }

    #[test]
    fn commitment_wire_is_not_claim_hash_domain() {
        assert_ne!(COMMITMENT_WIRE_DOMAIN, COMMITMENT_CLAIM_DOMAIN);
    }
}
