// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ASSURE-002: preregistered qualification campaigns.
//!
//! Governing theorem:
//!
//! ```text
//! plan content-addressed later
//!     != preregistered earlier
//!
//! semantic label frozen
//!     != semantic definition frozen
//!
//! self-declared timestamp
//!     != durable ordering evidence
//!
//! unique terminal receipt in supplied view
//!     != authoritative external currentness
//! ```
//!
//! This crate binds exact campaign semantics before admitted evidence begins.
//! It represents external monotonic ordering structurally but deliberately does
//! not authenticate the external ordering provider itself.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_assurance_core::{
    Claim, DigestSha256, EvidenceArtifact, EvidenceKind, QualificationPlan, StableId, SupportTier,
};
use symthaea_assurance_semantics::{
    SemanticCommitmentError, SemanticCommitmentV1, canonical_semantic_set,
};
use symthaea_assurance_subject::{AiSubjectManifest, SubjectError};
use thiserror::Error;

pub const ASSURE_CAMPAIGN_SCHEMA: &str = "symthaea.assurance.campaign.v1";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CampaignError {
    #[error("campaign plan requires at least one registered evidence kind")]
    EmptyEvidenceKinds,
    #[error("duplicate registered evidence kind: {0}")]
    DuplicateEvidenceKind(String),
    #[error("custom evidence kind requires an exact semantic definition commitment")]
    CustomEvidenceRequiresDefinition,
    #[error("duplicate semantic identifier in {set}: {id}")]
    DuplicateSemanticId { set: &'static str, id: String },
    #[error("support criterion exceeds campaign support ceiling")]
    SupportCriterionAboveCeiling,
    #[error("claim does not bind the supplied ASSURE-001 subject")]
    ClaimSubjectMismatch,
    #[error("ordering epoch must be non-zero")]
    ZeroOrderingEpoch,
    #[error("ordering sequence must be non-zero")]
    ZeroOrderingSequence,
    #[error("ordering receipt does not bind the expected statement")]
    OrderingStatementMismatch,
    #[error("ordering receipts are from incomparable source/profile/epoch lineages")]
    IncomparableOrderingLineage,
    #[error("ordering sequence is not strictly later than its predecessor")]
    NonIncreasingOrderingSequence,
    #[error("registration predecessor belongs to a different campaign")]
    CrossCampaignPredecessor,
    #[error("registration predecessor binds a different claim")]
    CrossClaimPredecessor,
    #[error("registration predecessor binds a different ASSURE-001 subject")]
    CrossSubjectPredecessor,
    #[error("registration set is empty")]
    EmptyRegistrationSet,
    #[error("registration set contains multiple roots")]
    MultipleRegistrationRoots,
    #[error("registration predecessor is missing from the supplied lineage")]
    MissingRegistrationPredecessor,
    #[error("registration lineage contains a fork")]
    RegistrationFork,
    #[error("registration lineage is disconnected")]
    DisconnectedRegistrationLineage,
    #[error("withdrawal targets an unknown registration")]
    UnknownWithdrawalTarget,
    #[error("conflicting withdrawals target the same registration")]
    ConflictingWithdrawal,
    #[error("a withdrawn registration has a successor")]
    SuccessorOfWithdrawnRegistration,
    #[error("the supplied registration view has no terminal unwithdrawn registration")]
    NoTerminalRegistrationInView,
    #[error("campaign plan does not match the terminal registration in the supplied view")]
    TerminalPlanMismatch,
    #[error("evidence subject does not match the exact campaign subject")]
    EvidenceSubjectMismatch,
    #[error("evidence claim does not match the exact campaign claim")]
    EvidenceClaimMismatch,
    #[error("evidence kind is not registered by the terminal campaign plan in the supplied view")]
    UnregisteredEvidenceKind,
    #[error(
        "evidence commitment statement does not bind the exact evidence/terminal registration in the supplied view"
    )]
    EvidenceCommitmentStatementMismatch,
    #[error(
        "evidence commitment is not after the terminal registration in the supplied view: {0:?}"
    )]
    EvidenceCommitmentNotAfterRegistration(EvidenceCommitmentTimingClass),
    #[error("evidence admission statement does not bind the exact current ledger state")]
    EvidenceAdmissionStatementMismatch,
    #[error("evidence admission ordering is not strictly later than evidence commitment ordering")]
    AdmissionNotAfterCommitment,
    #[error("evidence admission ordering is not strictly later than the ledger's prior admission")]
    AdmissionNotAfterLedgerHead,
    #[error("evidence ledger belongs to a different registration")]
    LedgerRegistrationMismatch,
    #[error("duplicate evidence id in the admitted campaign ledger: {0}")]
    DuplicateEvidenceId(String),
    #[error("duplicate evidence digest in the admitted campaign ledger")]
    DuplicateEvidenceDigest,
    #[error("evidence ordinal overflow")]
    EvidenceOrdinalOverflow,
    #[error(transparent)]
    Subject(#[from] SubjectError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceRequirementV1 {
    Builtin(EvidenceKind),
    Custom(SemanticCommitmentV1),
}

impl EvidenceRequirementV1 {
    pub fn builtin(kind: EvidenceKind) -> Result<Self, CampaignError> {
        if matches!(kind, EvidenceKind::Custom(_)) {
            return Err(CampaignError::CustomEvidenceRequiresDefinition);
        }
        Ok(Self::Builtin(kind))
    }

    pub fn custom(semantic: SemanticCommitmentV1) -> Self {
        Self::Custom(semantic)
    }

    pub fn core_kind(&self) -> EvidenceKind {
        match self {
            Self::Builtin(kind) => kind.clone(),
            Self::Custom(semantic) => EvidenceKind::Custom(semantic.semantic_id().clone()),
        }
    }

    fn canonical_name(&self) -> String {
        evidence_kind_name(&self.core_kind())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportCriterionV1 {
    tier: SupportTier,
    criterion: SemanticCommitmentV1,
}

impl SupportCriterionV1 {
    pub fn new(tier: SupportTier, criterion: SemanticCommitmentV1) -> Self {
        Self { tier, criterion }
    }

    pub fn tier(&self) -> SupportTier {
        self.tier
    }

    pub fn criterion(&self) -> &SemanticCommitmentV1 {
        &self.criterion
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReproductionRequirementV1 {
    NotRequired,
    DistinctVerifierEvidenceRequired,
}

impl ReproductionRequirementV1 {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::NotRequired => "not-required",
            Self::DistinctVerifierEvidenceRequired => "distinct-verifier-evidence-required",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignPlanV1 {
    plan_key: StableId,
    campaign_nonce: StableId,
    claim_digest: DigestSha256,
    subject_manifest_id: DigestSha256,
    subject_core_id: DigestSha256,
    maximum_support: SupportTier,
    reproduction_requirement: ReproductionRequirementV1,
    evidence_requirements: Vec<EvidenceRequirementV1>,
    support_criteria: Vec<SupportCriterionV1>,
    controls: Vec<SemanticCommitmentV1>,
    failure_conditions: Vec<SemanticCommitmentV1>,
    contradiction_conditions: Vec<SemanticCommitmentV1>,
    inconclusive_conditions: Vec<SemanticCommitmentV1>,
    invalidation_conditions: Vec<SemanticCommitmentV1>,
}

impl CampaignPlanV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plan_key: StableId,
        campaign_nonce: StableId,
        claim: &Claim,
        subject: &AiSubjectManifest,
        maximum_support: SupportTier,
        reproduction_requirement: ReproductionRequirementV1,
        evidence_requirements: Vec<EvidenceRequirementV1>,
        support_criteria: Vec<SupportCriterionV1>,
        controls: Vec<SemanticCommitmentV1>,
        failure_conditions: Vec<SemanticCommitmentV1>,
        contradiction_conditions: Vec<SemanticCommitmentV1>,
        inconclusive_conditions: Vec<SemanticCommitmentV1>,
        invalidation_conditions: Vec<SemanticCommitmentV1>,
    ) -> Result<Self, CampaignError> {
        let subject_core_id = subject.core_subject_id()?;
        if claim.subject_id() != &subject_core_id {
            return Err(CampaignError::ClaimSubjectMismatch);
        }

        let evidence_requirements = canonical_evidence_requirements(evidence_requirements)?;
        if evidence_requirements.is_empty() {
            return Err(CampaignError::EmptyEvidenceKinds);
        }
        let support_criteria = canonical_support_criteria(support_criteria, maximum_support)?;

        Ok(Self {
            plan_key,
            campaign_nonce,
            claim_digest: claim.digest(),
            subject_manifest_id: subject.manifest_id(),
            subject_core_id,
            maximum_support,
            reproduction_requirement,
            evidence_requirements,
            support_criteria,
            controls: canonical_semantic_set(controls)
                .map_err(|error| semantic_set_error("controls", error))?,
            failure_conditions: canonical_semantic_set(failure_conditions)
                .map_err(|error| semantic_set_error("failure-conditions", error))?,
            contradiction_conditions: canonical_semantic_set(contradiction_conditions)
                .map_err(|error| semantic_set_error("contradiction-conditions", error))?,
            inconclusive_conditions: canonical_semantic_set(inconclusive_conditions)
                .map_err(|error| semantic_set_error("inconclusive-conditions", error))?,
            invalidation_conditions: canonical_semantic_set(invalidation_conditions)
                .map_err(|error| semantic_set_error("invalidation-conditions", error))?,
        })
    }

    pub fn plan_key(&self) -> &StableId {
        &self.plan_key
    }

    pub fn campaign_nonce(&self) -> &StableId {
        &self.campaign_nonce
    }

    pub fn claim_digest(&self) -> &DigestSha256 {
        &self.claim_digest
    }

    pub fn subject_manifest_id(&self) -> &DigestSha256 {
        &self.subject_manifest_id
    }

    pub fn subject_core_id(&self) -> &DigestSha256 {
        &self.subject_core_id
    }

    pub fn maximum_support(&self) -> SupportTier {
        self.maximum_support
    }

    pub fn reproduction_requirement(&self) -> ReproductionRequirementV1 {
        self.reproduction_requirement
    }

    pub fn evidence_requirements(&self) -> &[EvidenceRequirementV1] {
        &self.evidence_requirements
    }

    pub fn support_criteria(&self) -> &[SupportCriterionV1] {
        &self.support_criteria
    }

    pub fn core_plan(&self) -> QualificationPlan {
        QualificationPlan::new(
            self.plan_key.clone(),
            self.claim_digest.clone(),
            self.maximum_support,
            self.invalidation_conditions
                .iter()
                .map(|condition| condition.semantic_id().clone())
                .collect(),
        )
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-campaign-plan-v1\n");
        field(&mut out, "schema", ASSURE_CAMPAIGN_SCHEMA);
        field(&mut out, "plan-key", self.plan_key.as_str());
        field(&mut out, "campaign-nonce", self.campaign_nonce.as_str());
        field(&mut out, "claim", self.claim_digest.as_str());
        field(
            &mut out,
            "assure-001-subject",
            self.subject_manifest_id.as_str(),
        );
        field(&mut out, "core-subject", self.subject_core_id.as_str());
        field(&mut out, "core-plan", self.core_plan().digest().as_str());
        field(
            &mut out,
            "maximum-support",
            support_tier_name(self.maximum_support),
        );
        field(
            &mut out,
            "reproduction-requirement",
            self.reproduction_requirement.canonical_name(),
        );
        append_evidence_requirements(&mut out, &self.evidence_requirements);
        append_support_criteria(&mut out, &self.support_criteria);
        append_semantics(&mut out, "control", &self.controls);
        append_semantics(&mut out, "failure-condition", &self.failure_conditions);
        append_semantics(
            &mut out,
            "contradiction-condition",
            &self.contradiction_conditions,
        );
        append_semantics(
            &mut out,
            "inconclusive-condition",
            &self.inconclusive_conditions,
        );
        append_semantics(
            &mut out,
            "invalidation-condition",
            &self.invalidation_conditions,
        );
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }

    fn registers_kind(&self, kind: &EvidenceKind) -> bool {
        self.evidence_requirements
            .iter()
            .any(|requirement| requirement.core_kind() == *kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderingReceiptV1 {
    source: StableId,
    validation_profile: SemanticCommitmentV1,
    epoch: u64,
    sequence: u64,
    statement_digest: DigestSha256,
    external_receipt_digest: DigestSha256,
}

impl OrderingReceiptV1 {
    pub fn new(
        source: StableId,
        validation_profile: SemanticCommitmentV1,
        epoch: u64,
        sequence: u64,
        statement_digest: DigestSha256,
        external_receipt_digest: DigestSha256,
    ) -> Result<Self, CampaignError> {
        if epoch == 0 {
            return Err(CampaignError::ZeroOrderingEpoch);
        }
        if sequence == 0 {
            return Err(CampaignError::ZeroOrderingSequence);
        }
        Ok(Self {
            source,
            validation_profile,
            epoch,
            sequence,
            statement_digest,
            external_receipt_digest,
        })
    }

    pub fn source(&self) -> &StableId {
        &self.source
    }

    pub fn validation_profile(&self) -> &SemanticCommitmentV1 {
        &self.validation_profile
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn statement_digest(&self) -> &DigestSha256 {
        &self.statement_digest
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-ordering-receipt-v1\n");
        field(&mut out, "source", self.source.as_str());
        field(
            &mut out,
            "validation-profile-id",
            self.validation_profile.semantic_id().as_str(),
        );
        field(
            &mut out,
            "validation-profile-commitment",
            self.validation_profile.digest().as_str(),
        );
        field(&mut out, "epoch", &canonical_u64(self.epoch));
        field(&mut out, "sequence", &canonical_u64(self.sequence));
        field(&mut out, "statement", self.statement_digest.as_str());
        field(
            &mut out,
            "external-receipt",
            self.external_receipt_digest.as_str(),
        );
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }

    fn require_statement(&self, expected: &DigestSha256) -> Result<(), CampaignError> {
        if &self.statement_digest == expected {
            Ok(())
        } else {
            Err(CampaignError::OrderingStatementMismatch)
        }
    }

    fn require_later_than(&self, earlier: &Self) -> Result<(), CampaignError> {
        if self.source != earlier.source
            || self.validation_profile != earlier.validation_profile
            || self.epoch != earlier.epoch
        {
            return Err(CampaignError::IncomparableOrderingLineage);
        }
        if self.sequence <= earlier.sequence {
            return Err(CampaignError::NonIncreasingOrderingSequence);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrationStatementV1 {
    plan_digest: DigestSha256,
    campaign_nonce: StableId,
    claim_digest: DigestSha256,
    subject_manifest_id: DigestSha256,
    issuer: StableId,
    predecessor_receipt: Option<DigestSha256>,
    empty_evidence_root: DigestSha256,
}

impl RegistrationStatementV1 {
    pub fn new(
        plan: &CampaignPlanV1,
        issuer: StableId,
        predecessor: Option<&PreregistrationReceiptV1>,
    ) -> Result<Self, CampaignError> {
        if let Some(previous) = predecessor {
            if previous.campaign_nonce() != plan.campaign_nonce() {
                return Err(CampaignError::CrossCampaignPredecessor);
            }
            if previous.claim_digest() != plan.claim_digest() {
                return Err(CampaignError::CrossClaimPredecessor);
            }
            if previous.subject_manifest_id() != plan.subject_manifest_id() {
                return Err(CampaignError::CrossSubjectPredecessor);
            }
        }

        let predecessor_receipt = predecessor.map(PreregistrationReceiptV1::digest);
        let empty_evidence_root = empty_evidence_root(
            plan.campaign_nonce(),
            &plan.digest(),
            predecessor_receipt.as_ref(),
        );

        Ok(Self {
            plan_digest: plan.digest(),
            campaign_nonce: plan.campaign_nonce.clone(),
            claim_digest: plan.claim_digest.clone(),
            subject_manifest_id: plan.subject_manifest_id.clone(),
            issuer,
            predecessor_receipt,
            empty_evidence_root,
        })
    }

    pub fn plan_digest(&self) -> &DigestSha256 {
        &self.plan_digest
    }

    pub fn campaign_nonce(&self) -> &StableId {
        &self.campaign_nonce
    }

    pub fn predecessor_receipt(&self) -> Option<&DigestSha256> {
        self.predecessor_receipt.as_ref()
    }

    pub fn empty_evidence_root(&self) -> &DigestSha256 {
        &self.empty_evidence_root
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-registration-statement-v1\n");
        field(&mut out, "plan", self.plan_digest.as_str());
        field(&mut out, "campaign-nonce", self.campaign_nonce.as_str());
        field(&mut out, "claim", self.claim_digest.as_str());
        field(
            &mut out,
            "assure-001-subject",
            self.subject_manifest_id.as_str(),
        );
        field(&mut out, "issuer", self.issuer.as_str());
        optional_digest(
            &mut out,
            "predecessor-registration",
            self.predecessor_receipt.as_ref(),
        );
        field(&mut out, "pre-evidence-count", "0");
        field(
            &mut out,
            "pre-evidence-root",
            self.empty_evidence_root.as_str(),
        );
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreregistrationReceiptV1 {
    statement: RegistrationStatementV1,
    ordering: OrderingReceiptV1,
}

impl PreregistrationReceiptV1 {
    pub fn new(
        statement: RegistrationStatementV1,
        ordering: OrderingReceiptV1,
        predecessor: Option<&PreregistrationReceiptV1>,
    ) -> Result<Self, CampaignError> {
        ordering.require_statement(&statement.digest())?;

        match (statement.predecessor_receipt(), predecessor) {
            (None, None) => {}
            (Some(expected), Some(previous)) if expected == &previous.digest() => {
                ordering.require_later_than(&previous.ordering)?;
            }
            _ => return Err(CampaignError::MissingRegistrationPredecessor),
        }

        Ok(Self {
            statement,
            ordering,
        })
    }

    pub fn plan_digest(&self) -> &DigestSha256 {
        self.statement.plan_digest()
    }

    pub fn campaign_nonce(&self) -> &StableId {
        self.statement.campaign_nonce()
    }

    pub fn claim_digest(&self) -> &DigestSha256 {
        &self.statement.claim_digest
    }

    pub fn subject_manifest_id(&self) -> &DigestSha256 {
        &self.statement.subject_manifest_id
    }

    pub fn predecessor_receipt(&self) -> Option<&DigestSha256> {
        self.statement.predecessor_receipt()
    }

    pub fn ordering(&self) -> &OrderingReceiptV1 {
        &self.ordering
    }

    pub fn empty_evidence_root(&self) -> &DigestSha256 {
        self.statement.empty_evidence_root()
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-preregistration-receipt-v1\n");
        field(&mut out, "statement", self.statement.digest().as_str());
        field(&mut out, "ordering", self.ordering.digest().as_str());
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WithdrawalStatementV1 {
    campaign_nonce: StableId,
    target_registration: DigestSha256,
    reason: StableId,
}

impl WithdrawalStatementV1 {
    pub fn new(target: &PreregistrationReceiptV1, reason: StableId) -> Self {
        Self {
            campaign_nonce: target.campaign_nonce().clone(),
            target_registration: target.digest(),
            reason,
        }
    }

    pub fn digest(&self) -> DigestSha256 {
        let mut out = String::from("symthaea-assurance-registration-withdrawal-statement-v1\n");
        field(&mut out, "campaign-nonce", self.campaign_nonce.as_str());
        field(
            &mut out,
            "target-registration",
            self.target_registration.as_str(),
        );
        field(&mut out, "reason", self.reason.as_str());
        digest_canonical(out.as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrationWithdrawalV1 {
    statement: WithdrawalStatementV1,
    ordering: OrderingReceiptV1,
}

impl RegistrationWithdrawalV1 {
    pub fn new(
        statement: WithdrawalStatementV1,
        ordering: OrderingReceiptV1,
        target: &PreregistrationReceiptV1,
    ) -> Result<Self, CampaignError> {
        ordering.require_statement(&statement.digest())?;
        if statement.target_registration != target.digest()
            || &statement.campaign_nonce != target.campaign_nonce()
        {
            return Err(CampaignError::UnknownWithdrawalTarget);
        }
        ordering.require_later_than(target.ordering())?;
        Ok(Self {
            statement,
            ordering,
        })
    }

    pub fn target_registration(&self) -> &DigestSha256 {
        &self.statement.target_registration
    }

    pub fn ordering(&self) -> &OrderingReceiptV1 {
        &self.ordering
    }

    pub fn digest(&self) -> DigestSha256 {
        let mut out = String::from("symthaea-assurance-registration-withdrawal-v1\n");
        field(&mut out, "statement", self.statement.digest().as_str());
        field(&mut out, "ordering", self.ordering.digest().as_str());
        digest_canonical(out.as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalRegistrationInViewV1 {
    receipt: PreregistrationReceiptV1,
}

impl TerminalRegistrationInViewV1 {
    pub fn receipt(&self) -> &PreregistrationReceiptV1 {
        &self.receipt
    }

    pub fn digest(&self) -> DigestSha256 {
        self.receipt.digest()
    }
}

pub fn resolve_terminal_registration_in_view(
    registrations: &[PreregistrationReceiptV1],
    withdrawals: &[RegistrationWithdrawalV1],
) -> Result<TerminalRegistrationInViewV1, CampaignError> {
    if registrations.is_empty() {
        return Err(CampaignError::EmptyRegistrationSet);
    }

    let mut by_digest = BTreeMap::<DigestSha256, PreregistrationReceiptV1>::new();
    for registration in registrations {
        let digest = registration.digest();
        if let Some(existing) = by_digest.get(&digest) {
            if existing.canonical_bytes() != registration.canonical_bytes() {
                return Err(CampaignError::RegistrationFork);
            }
            continue;
        }
        by_digest.insert(digest, registration.clone());
    }

    let roots: Vec<_> = by_digest
        .iter()
        .filter(|(_, registration)| registration.predecessor_receipt().is_none())
        .map(|(digest, _)| digest.clone())
        .collect();
    if roots.len() != 1 {
        return Err(CampaignError::MultipleRegistrationRoots);
    }

    let root_campaign = by_digest[&roots[0]].campaign_nonce().clone();
    let mut children = BTreeMap::<DigestSha256, Vec<DigestSha256>>::new();
    for (digest, registration) in &by_digest {
        if registration.campaign_nonce() != &root_campaign {
            return Err(CampaignError::DisconnectedRegistrationLineage);
        }
        if let Some(predecessor) = registration.predecessor_receipt() {
            let previous = by_digest
                .get(predecessor)
                .ok_or(CampaignError::MissingRegistrationPredecessor)?;
            registration
                .ordering()
                .require_later_than(previous.ordering())?;
            let list = children.entry(predecessor.clone()).or_default();
            list.push(digest.clone());
            if list.len() > 1 {
                return Err(CampaignError::RegistrationFork);
            }
        }
    }

    let mut withdrawal_by_target = BTreeMap::<DigestSha256, RegistrationWithdrawalV1>::new();
    for withdrawal in withdrawals {
        let target = by_digest
            .get(withdrawal.target_registration())
            .ok_or(CampaignError::UnknownWithdrawalTarget)?;
        withdrawal
            .ordering()
            .require_later_than(target.ordering())?;
        if let Some(existing) = withdrawal_by_target.get(withdrawal.target_registration()) {
            if existing.digest() != withdrawal.digest() {
                return Err(CampaignError::ConflictingWithdrawal);
            }
        } else {
            withdrawal_by_target
                .insert(withdrawal.target_registration().clone(), withdrawal.clone());
        }
    }

    let mut current = roots[0].clone();
    let mut visited = BTreeSet::new();
    loop {
        visited.insert(current.clone());
        match children.get(&current) {
            Some(next) if next.len() == 1 => {
                if withdrawal_by_target.contains_key(&current) {
                    return Err(CampaignError::SuccessorOfWithdrawnRegistration);
                }
                current = next[0].clone();
            }
            Some(_) => return Err(CampaignError::RegistrationFork),
            None => break,
        }
    }

    if visited.len() != by_digest.len() {
        return Err(CampaignError::DisconnectedRegistrationLineage);
    }
    if withdrawal_by_target.contains_key(&current) {
        return Err(CampaignError::NoTerminalRegistrationInView);
    }

    Ok(TerminalRegistrationInViewV1 {
        receipt: by_digest[&current].clone(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceCommitmentTimingClass {
    CommittedAfterTerminalRegistrationInView,
    CommittedBeforeOrAtTerminalRegistrationInView,
    IncomparableOrderingLineage,
}

pub fn evidence_commitment_statement_digest(
    current: &TerminalRegistrationInViewV1,
    evidence: &EvidenceArtifact,
) -> DigestSha256 {
    let mut out = String::from("symthaea-assurance-evidence-commitment-statement-v1\n");
    field(&mut out, "registration", current.receipt.digest().as_str());
    field(
        &mut out,
        "campaign-nonce",
        current.receipt.campaign_nonce().as_str(),
    );
    field(&mut out, "plan", current.receipt.plan_digest().as_str());
    field(&mut out, "evidence", evidence.digest().as_str());
    digest_canonical(out.as_bytes())
}

pub fn classify_evidence_commitment_timing(
    current: &TerminalRegistrationInViewV1,
    evidence: &EvidenceArtifact,
    commitment_ordering: &OrderingReceiptV1,
) -> Result<EvidenceCommitmentTimingClass, CampaignError> {
    if commitment_ordering.statement_digest()
        != &evidence_commitment_statement_digest(current, evidence)
    {
        return Err(CampaignError::EvidenceCommitmentStatementMismatch);
    }

    let registration = current.receipt.ordering();
    if commitment_ordering.source() != registration.source()
        || commitment_ordering.validation_profile() != registration.validation_profile()
        || commitment_ordering.epoch() != registration.epoch()
    {
        return Ok(EvidenceCommitmentTimingClass::IncomparableOrderingLineage);
    }
    if commitment_ordering.sequence() <= registration.sequence() {
        return Ok(EvidenceCommitmentTimingClass::CommittedBeforeOrAtTerminalRegistrationInView);
    }
    Ok(EvidenceCommitmentTimingClass::CommittedAfterTerminalRegistrationInView)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignEvidenceAdmissionV1 {
    registration_digest: DigestSha256,
    plan_digest: DigestSha256,
    campaign_nonce: StableId,
    ordinal: u64,
    evidence_digest: DigestSha256,
    commitment_ordering_digest: DigestSha256,
    admission_ordering_digest: DigestSha256,
    previous_evidence_root: DigestSha256,
    evidence_root: DigestSha256,
}

impl CampaignEvidenceAdmissionV1 {
    pub fn evidence_root(&self) -> &DigestSha256 {
        &self.evidence_root
    }

    pub fn digest(&self) -> DigestSha256 {
        let mut out = String::from("symthaea-assurance-campaign-evidence-admission-v1\n");
        field(&mut out, "registration", self.registration_digest.as_str());
        field(&mut out, "plan", self.plan_digest.as_str());
        field(&mut out, "campaign-nonce", self.campaign_nonce.as_str());
        field(&mut out, "ordinal", &canonical_u64(self.ordinal));
        field(&mut out, "evidence", self.evidence_digest.as_str());
        field(
            &mut out,
            "commitment-ordering",
            self.commitment_ordering_digest.as_str(),
        );
        field(
            &mut out,
            "admission-ordering",
            self.admission_ordering_digest.as_str(),
        );
        field(
            &mut out,
            "previous-evidence-root",
            self.previous_evidence_root.as_str(),
        );
        field(&mut out, "evidence-root", self.evidence_root.as_str());
        digest_canonical(out.as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignEvidenceLedgerV1 {
    registration_digest: DigestSha256,
    campaign_nonce: StableId,
    plan_digest: DigestSha256,
    evidence_root: DigestSha256,
    admitted_count: u64,
    last_admission: OrderingReceiptV1,
    seen_evidence_ids: BTreeSet<StableId>,
    seen_evidence: BTreeSet<DigestSha256>,
}

impl CampaignEvidenceLedgerV1 {
    pub fn new(current: &TerminalRegistrationInViewV1) -> Self {
        Self {
            registration_digest: current.receipt.digest(),
            campaign_nonce: current.receipt.campaign_nonce().clone(),
            plan_digest: current.receipt.plan_digest().clone(),
            evidence_root: current.receipt.empty_evidence_root().clone(),
            admitted_count: 0,
            last_admission: current.receipt.ordering().clone(),
            seen_evidence_ids: BTreeSet::new(),
            seen_evidence: BTreeSet::new(),
        }
    }

    pub fn evidence_root(&self) -> &DigestSha256 {
        &self.evidence_root
    }

    pub fn admitted_count(&self) -> u64 {
        self.admitted_count
    }

    pub fn admission_statement_digest(
        &self,
        current: &TerminalRegistrationInViewV1,
        evidence: &EvidenceArtifact,
        commitment_ordering: &OrderingReceiptV1,
    ) -> Result<DigestSha256, CampaignError> {
        self.require_current(current)?;
        let ordinal = checked_next_evidence_ordinal(self.admitted_count)?;
        let mut out = String::from("symthaea-assurance-evidence-admission-statement-v1\n");
        field(&mut out, "registration", self.registration_digest.as_str());
        field(&mut out, "campaign-nonce", self.campaign_nonce.as_str());
        field(&mut out, "plan", self.plan_digest.as_str());
        field(&mut out, "ordinal", &canonical_u64(ordinal));
        field(&mut out, "evidence", evidence.digest().as_str());
        field(&mut out, "prior-root", self.evidence_root.as_str());
        field(
            &mut out,
            "previous-admission-ordering",
            self.last_admission.digest().as_str(),
        );
        field(
            &mut out,
            "commitment-ordering",
            commitment_ordering.digest().as_str(),
        );
        Ok(digest_canonical(out.as_bytes()))
    }

    pub fn admit_preregistered(
        &mut self,
        plan: &CampaignPlanV1,
        current: &TerminalRegistrationInViewV1,
        evidence: &EvidenceArtifact,
        commitment_ordering: &OrderingReceiptV1,
        admission: &OrderingReceiptV1,
    ) -> Result<CampaignEvidenceAdmissionV1, CampaignError> {
        self.require_current(current)?;
        if plan.digest() != self.plan_digest
            || plan.campaign_nonce() != &self.campaign_nonce
            || current.receipt.plan_digest() != &plan.digest()
        {
            return Err(CampaignError::TerminalPlanMismatch);
        }
        if evidence.subject_id() != plan.subject_core_id() {
            return Err(CampaignError::EvidenceSubjectMismatch);
        }
        if evidence.claim_digest() != plan.claim_digest() {
            return Err(CampaignError::EvidenceClaimMismatch);
        }
        if !plan.registers_kind(evidence.kind()) {
            return Err(CampaignError::UnregisteredEvidenceKind);
        }
        if self.seen_evidence_ids.contains(evidence.evidence_id()) {
            return Err(CampaignError::DuplicateEvidenceId(
                evidence.evidence_id().as_str().to_owned(),
            ));
        }

        let evidence_digest = evidence.digest();
        if self.seen_evidence.contains(&evidence_digest) {
            return Err(CampaignError::DuplicateEvidenceDigest);
        }

        let timing = classify_evidence_commitment_timing(current, evidence, commitment_ordering)?;
        if timing != EvidenceCommitmentTimingClass::CommittedAfterTerminalRegistrationInView {
            return Err(CampaignError::EvidenceCommitmentNotAfterRegistration(
                timing,
            ));
        }

        let expected_admission =
            self.admission_statement_digest(current, evidence, commitment_ordering)?;
        if admission.statement_digest() != &expected_admission {
            return Err(CampaignError::EvidenceAdmissionStatementMismatch);
        }
        admission
            .require_later_than(commitment_ordering)
            .map_err(|error| match error {
                CampaignError::NonIncreasingOrderingSequence => {
                    CampaignError::AdmissionNotAfterCommitment
                }
                other => other,
            })?;
        admission
            .require_later_than(&self.last_admission)
            .map_err(|error| match error {
                CampaignError::NonIncreasingOrderingSequence => {
                    CampaignError::AdmissionNotAfterLedgerHead
                }
                other => other,
            })?;

        let ordinal = checked_next_evidence_ordinal(self.admitted_count)?;
        let previous_evidence_root = self.evidence_root.clone();
        let evidence_root = next_evidence_root(
            &previous_evidence_root,
            ordinal,
            &evidence_digest,
            &commitment_ordering.digest(),
            &admission.digest(),
        );
        let result = CampaignEvidenceAdmissionV1 {
            registration_digest: self.registration_digest.clone(),
            plan_digest: self.plan_digest.clone(),
            campaign_nonce: self.campaign_nonce.clone(),
            ordinal,
            evidence_digest: evidence_digest.clone(),
            commitment_ordering_digest: commitment_ordering.digest(),
            admission_ordering_digest: admission.digest(),
            previous_evidence_root,
            evidence_root: evidence_root.clone(),
        };

        self.seen_evidence_ids
            .insert(evidence.evidence_id().clone());
        self.seen_evidence.insert(evidence_digest);
        self.admitted_count = ordinal;
        self.evidence_root = evidence_root;
        self.last_admission = admission.clone();
        Ok(result)
    }

    fn require_current(&self, current: &TerminalRegistrationInViewV1) -> Result<(), CampaignError> {
        if self.registration_digest != current.receipt.digest()
            || &self.campaign_nonce != current.receipt.campaign_nonce()
            || &self.plan_digest != current.receipt.plan_digest()
        {
            return Err(CampaignError::LedgerRegistrationMismatch);
        }
        Ok(())
    }
}

fn canonical_evidence_requirements(
    mut requirements: Vec<EvidenceRequirementV1>,
) -> Result<Vec<EvidenceRequirementV1>, CampaignError> {
    requirements.sort_by_cached_key(EvidenceRequirementV1::canonical_name);
    for pair in requirements.windows(2) {
        if pair[0].core_kind() == pair[1].core_kind() {
            return Err(CampaignError::DuplicateEvidenceKind(
                pair[0].canonical_name(),
            ));
        }
    }
    Ok(requirements)
}

fn canonical_support_criteria(
    mut criteria: Vec<SupportCriterionV1>,
    maximum_support: SupportTier,
) -> Result<Vec<SupportCriterionV1>, CampaignError> {
    for criterion in &criteria {
        if support_tier_rank(criterion.tier) > support_tier_rank(maximum_support) {
            return Err(CampaignError::SupportCriterionAboveCeiling);
        }
    }
    criteria.sort_by(|left, right| {
        support_tier_rank(left.tier)
            .cmp(&support_tier_rank(right.tier))
            .then_with(|| {
                left.criterion
                    .semantic_id()
                    .cmp(right.criterion.semantic_id())
            })
    });
    let mut ids = BTreeSet::new();
    for criterion in &criteria {
        if !ids.insert(criterion.criterion.semantic_id().clone()) {
            return Err(CampaignError::DuplicateSemanticId {
                set: "support-criteria",
                id: criterion.criterion.semantic_id().as_str().to_owned(),
            });
        }
    }
    Ok(criteria)
}

fn semantic_set_error(set: &'static str, error: SemanticCommitmentError) -> CampaignError {
    match error {
        SemanticCommitmentError::DuplicateSemanticId(id) => {
            CampaignError::DuplicateSemanticId { set, id }
        }
    }
}

fn checked_next_evidence_ordinal(current: u64) -> Result<u64, CampaignError> {
    current
        .checked_add(1)
        .ok_or(CampaignError::EvidenceOrdinalOverflow)
}

fn canonical_u64(value: u64) -> String {
    value.to_string()
}

fn empty_evidence_root(
    campaign_nonce: &StableId,
    plan_digest: &DigestSha256,
    predecessor: Option<&DigestSha256>,
) -> DigestSha256 {
    let mut out = String::from("symthaea-assurance-empty-evidence-root-v1\n");
    field(&mut out, "campaign-nonce", campaign_nonce.as_str());
    field(&mut out, "plan", plan_digest.as_str());
    optional_digest(&mut out, "predecessor-registration", predecessor);
    digest_canonical(out.as_bytes())
}

fn next_evidence_root(
    previous: &DigestSha256,
    ordinal: u64,
    evidence: &DigestSha256,
    commitment_ordering: &DigestSha256,
    admission_ordering: &DigestSha256,
) -> DigestSha256 {
    let mut out = String::from("symthaea-assurance-evidence-root-step-v1\n");
    field(&mut out, "previous", previous.as_str());
    field(&mut out, "ordinal", &canonical_u64(ordinal));
    field(&mut out, "evidence", evidence.as_str());
    field(
        &mut out,
        "commitment-ordering",
        commitment_ordering.as_str(),
    );
    field(&mut out, "admission-ordering", admission_ordering.as_str());
    digest_canonical(out.as_bytes())
}

fn append_evidence_requirements(out: &mut String, requirements: &[EvidenceRequirementV1]) {
    field(out, "evidence-kind-count", &requirements.len().to_string());
    for requirement in requirements {
        field(out, "evidence-kind", &requirement.canonical_name());
        match requirement {
            EvidenceRequirementV1::Builtin(_) => {
                field(out, "evidence-kind-semantic-commitment", "")
            }
            EvidenceRequirementV1::Custom(semantic) => field(
                out,
                "evidence-kind-semantic-commitment",
                semantic.digest().as_str(),
            ),
        }
    }
}

fn append_support_criteria(out: &mut String, criteria: &[SupportCriterionV1]) {
    field(out, "support-criterion-count", &criteria.len().to_string());
    for criterion in criteria {
        field(
            out,
            "support-criterion-tier",
            support_tier_name(criterion.tier),
        );
        field(
            out,
            "support-criterion-id",
            criterion.criterion.semantic_id().as_str(),
        );
        field(
            out,
            "support-criterion-commitment",
            criterion.criterion.digest().as_str(),
        );
    }
}

fn append_semantics(out: &mut String, label: &str, values: &[SemanticCommitmentV1]) {
    field(out, &format!("{label}-count"), &values.len().to_string());
    for value in values {
        field(out, &format!("{label}-id"), value.semantic_id().as_str());
        field(out, &format!("{label}-commitment"), value.digest().as_str());
    }
}

fn support_tier_rank(tier: SupportTier) -> u8 {
    match tier {
        SupportTier::Structural => 0,
        SupportTier::Observed => 1,
        SupportTier::CausallySupported => 2,
        SupportTier::FunctionallySupported => 3,
    }
}

fn support_tier_name(tier: SupportTier) -> &'static str {
    match tier {
        SupportTier::Structural => "structural",
        SupportTier::Observed => "observed",
        SupportTier::CausallySupported => "causally-supported",
        SupportTier::FunctionallySupported => "functionally-supported",
    }
}

fn evidence_kind_name(kind: &EvidenceKind) -> String {
    match kind {
        EvidenceKind::ArchitectureInspection => "architecture-inspection".into(),
        EvidenceKind::Observation => "observation".into(),
        EvidenceKind::ControlledIntervention => "controlled-intervention".into(),
        EvidenceKind::FunctionalBenchmark => "functional-benchmark".into(),
        EvidenceKind::Reproduction => "reproduction".into(),
        EvidenceKind::RuntimeReceipt => "runtime-receipt".into(),
        EvidenceKind::ExternalAttestation => "external-attestation".into(),
        EvidenceKind::Custom(id) => format!("custom:{}", id.as_str()),
    }
}

fn optional_digest(out: &mut String, label: &str, value: Option<&DigestSha256>) {
    field(out, label, value.map(DigestSha256::as_str).unwrap_or(""));
}

fn field(out: &mut String, label: &str, value: &str) {
    out.push_str(label);
    out.push(' ');
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
    out.push('\n');
}

fn digest_canonical(bytes: &[u8]) -> DigestSha256 {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    DigestSha256::new(encoded).expect("SHA-256 encoding is always valid")
}

#[cfg(test)]
mod convergence_tests {
    use super::*;

    #[test]
    fn evidence_ordinal_overflow_fails_closed() {
        assert_eq!(
            checked_next_evidence_ordinal(u64::MAX),
            Err(CampaignError::EvidenceOrdinalOverflow)
        );
    }

    #[test]
    fn canonical_u64_is_minimal_unsigned_decimal_ascii() {
        assert_eq!(canonical_u64(0), "0");
        assert_eq!(canonical_u64(7), "7");
        assert_eq!(canonical_u64(u64::MAX), "18446744073709551615");
    }
}
