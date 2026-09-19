// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authenticated freshness assessment for a qualified claim's frozen evidence surface.
//!
//! A freshness pass is not a new search strategy. It re-runs the exact source,
//! scope, query, retrieval, decision, and deduplication semantics used by the
//! baseline qualification against later authenticated source snapshots. A changed
//! search strategy is a new evidence protocol and cannot masquerade as freshness.
//!
//! A positive result establishes freshness only within that frozen protocol. It
//! does not establish global evidence exhaustiveness, current scientific validity,
//! or scientific truth.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use symthaea_trust_core::{
    AuthorizedTrustRoleAttestation, FramedDigest, RootRoleQuorumProof,
    Sha256Digest as TrustSha256Digest, TrustRole, TrustedTime,
};

use crate::{
    ClaimRelationBindingReport, CorpusItemDecision, EvidenceCoverageAuthorityReport,
    EvidenceCoverageClosure, EvidenceDecisionBindingClosure, EvidenceDecisionBindingReport,
    EvidenceRecord, EvidenceSourceSpec, FrozenEvidenceCoverageProtocol, QualifiedScientificClaim,
    ResearchId, RetrievalOutcome, ScientificClaim, Sha256Digest, SourceRetrievalReceipt,
    audit_claim_evidence_coverage,
};

const EVIDENCE_SOURCE_FRESHNESS_STATEMENT_DOMAIN: &str =
    "symthaea.evidence-source-freshness-statement.identity.v1";
const AUTHENTICATED_EVIDENCE_SOURCE_FRESHNESS_DOMAIN: &str =
    "symthaea.authenticated-evidence-source-freshness.identity.v1";
const EVIDENCE_FRESHNESS_ASSESSMENT_DOMAIN: &str =
    "symthaea.scientific-evidence-freshness-assessment.identity.v1";
pub const MAX_FRESHNESS_SOURCES: usize = 4096;

/// Authority-independent statement that one exact source snapshot/retrieval was
/// observed through a declared cutoff. Authority is established separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceSourceFreshnessStatement {
    source_id: ResearchId,
    source_scope_sha256: Sha256Digest,
    query_sha256: Sha256Digest,
    retrieval_protocol_sha256: Sha256Digest,
    index_snapshot_sha256: Sha256Digest,
    retrieval_artifact_sha256: Sha256Digest,
    observed_through_unix_ms: u64,
    statement_sha256: TrustSha256Digest,
}

impl EvidenceSourceFreshnessStatement {
    pub fn new(
        source: &EvidenceSourceSpec,
        receipt: &SourceRetrievalReceipt,
        observed_through_unix_ms: u64,
    ) -> Self {
        let statement_sha256 = source_freshness_statement_digest(
            source,
            receipt,
            observed_through_unix_ms,
        );
        Self {
            source_id: source.source_id.clone(),
            source_scope_sha256: source.source_scope_sha256.clone(),
            query_sha256: source.query_sha256.clone(),
            retrieval_protocol_sha256: source.retrieval_protocol_sha256.clone(),
            index_snapshot_sha256: receipt.index_snapshot_sha256.clone(),
            retrieval_artifact_sha256: receipt.retrieval_artifact_sha256.clone(),
            observed_through_unix_ms,
            statement_sha256,
        }
    }

    pub fn source_id(&self) -> &ResearchId { &self.source_id }
    pub fn observed_through_unix_ms(&self) -> u64 { self.observed_through_unix_ms }
    pub fn statement_sha256(&self) -> &TrustSha256Digest { &self.statement_sha256 }
    pub const fn freshness_authority_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceSourceFreshnessAuthenticationError {
    UnknownSource,
    SourceIdentityMismatch,
    SourceScopeMismatch,
    QueryMismatch,
    RetrievalProtocolMismatch,
    IndexSnapshotMismatch,
    RetrievalArtifactMismatch,
    RetrievalNotComplete,
    ObservedThroughBeforeSearchCutoff,
    WrongRole,
    RootAuthorityMismatch,
    SubjectMismatch,
    PayloadMismatch,
    ContextMismatch,
    RoleProofWrongRole,
    RoleProofMismatch,
    RoleProofOutsideTrustedTime,
    ObservedThroughAfterAttestation,
    TimeDependsOnSourceSnapshot,
}

/// Non-forgeable authority that one exact evidence-source snapshot/retrieval was
/// fresh through the stated cutoff under the root's generic Freshness role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthenticatedEvidenceSourceFreshness {
    source_id: ResearchId,
    coverage_protocol_sha256: Sha256Digest,
    index_snapshot_sha256: Sha256Digest,
    retrieval_artifact_sha256: Sha256Digest,
    observed_through_unix_ms: u64,
    root_authority_sha256: TrustSha256Digest,
    trust_snapshot_authority_sha256: TrustSha256Digest,
    freshness_role_authority_sha256: TrustSha256Digest,
    role_quorum_proof_sha256: TrustSha256Digest,
    source_freshness_sha256: TrustSha256Digest,
}

impl AuthenticatedEvidenceSourceFreshness {
    pub fn source_id(&self) -> &ResearchId { &self.source_id }
    pub fn coverage_protocol_sha256(&self) -> &Sha256Digest { &self.coverage_protocol_sha256 }
    pub fn index_snapshot_sha256(&self) -> &Sha256Digest { &self.index_snapshot_sha256 }
    pub fn retrieval_artifact_sha256(&self) -> &Sha256Digest { &self.retrieval_artifact_sha256 }
    pub fn observed_through_unix_ms(&self) -> u64 { self.observed_through_unix_ms }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &TrustSha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn freshness_role_authority_sha256(&self) -> &TrustSha256Digest {
        &self.freshness_role_authority_sha256
    }
    pub fn role_quorum_proof_sha256(&self) -> &TrustSha256Digest {
        &self.role_quorum_proof_sha256
    }
    pub fn source_freshness_sha256(&self) -> &TrustSha256Digest { &self.source_freshness_sha256 }
    pub const fn source_freshness_authority_established(&self) -> bool { true }
    pub const fn global_source_exhaustiveness_established(&self) -> bool { false }
}

pub fn authenticate_evidence_source_freshness(
    frozen_protocol: &FrozenEvidenceCoverageProtocol,
    receipt: &SourceRetrievalReceipt,
    statement: &EvidenceSourceFreshnessStatement,
    trusted_time: &TrustedTime,
    freshness_authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> Result<AuthenticatedEvidenceSourceFreshness, EvidenceSourceFreshnessAuthenticationError> {
    let Some(source) = frozen_protocol
        .protocol()
        .sources
        .iter()
        .find(|source| source.source_id == receipt.source_id)
    else {
        return Err(EvidenceSourceFreshnessAuthenticationError::UnknownSource);
    };
    if statement.source_id != source.source_id || receipt.source_id != source.source_id {
        return Err(EvidenceSourceFreshnessAuthenticationError::SourceIdentityMismatch);
    }
    if statement.source_scope_sha256 != source.source_scope_sha256
        || receipt.source_scope_sha256 != source.source_scope_sha256
    {
        return Err(EvidenceSourceFreshnessAuthenticationError::SourceScopeMismatch);
    }
    if statement.query_sha256 != source.query_sha256 || receipt.query_sha256 != source.query_sha256 {
        return Err(EvidenceSourceFreshnessAuthenticationError::QueryMismatch);
    }
    if statement.retrieval_protocol_sha256 != source.retrieval_protocol_sha256
        || receipt.retrieval_protocol_sha256 != source.retrieval_protocol_sha256
    {
        return Err(EvidenceSourceFreshnessAuthenticationError::RetrievalProtocolMismatch);
    }
    if statement.index_snapshot_sha256 != receipt.index_snapshot_sha256 {
        return Err(EvidenceSourceFreshnessAuthenticationError::IndexSnapshotMismatch);
    }
    if statement.retrieval_artifact_sha256 != receipt.retrieval_artifact_sha256 {
        return Err(EvidenceSourceFreshnessAuthenticationError::RetrievalArtifactMismatch);
    }
    if receipt.outcome != RetrievalOutcome::Complete {
        return Err(EvidenceSourceFreshnessAuthenticationError::RetrievalNotComplete);
    }
    if statement.observed_through_unix_ms < frozen_protocol.protocol().declared_search_cutoff_unix_ms {
        return Err(EvidenceSourceFreshnessAuthenticationError::ObservedThroughBeforeSearchCutoff);
    }
    if freshness_authority.role() != TrustRole::Freshness {
        return Err(EvidenceSourceFreshnessAuthenticationError::WrongRole);
    }
    if freshness_authority.root_authority_sha256() != trusted_time.root_authority_sha256() {
        return Err(EvidenceSourceFreshnessAuthenticationError::RootAuthorityMismatch);
    }
    if freshness_authority.subject_sha256() != &bridge_digest(&receipt.index_snapshot_sha256) {
        return Err(EvidenceSourceFreshnessAuthenticationError::SubjectMismatch);
    }
    if freshness_authority.payload_sha256() != statement.statement_sha256() {
        return Err(EvidenceSourceFreshnessAuthenticationError::PayloadMismatch);
    }
    let retrieval_artifact = bridge_digest(&receipt.retrieval_artifact_sha256);
    if freshness_authority.context_sha256() != Some(&retrieval_artifact) {
        return Err(EvidenceSourceFreshnessAuthenticationError::ContextMismatch);
    }
    if role_proof.role() != TrustRole::Freshness {
        return Err(EvidenceSourceFreshnessAuthenticationError::RoleProofWrongRole);
    }
    if freshness_authority.role_quorum_proof_sha256() != role_proof.proof_sha256() {
        return Err(EvidenceSourceFreshnessAuthenticationError::RoleProofMismatch);
    }
    let (time_earliest, time_latest) = trusted_time.consensus_interval();
    if role_proof.evaluation_time_unix_s() < time_earliest
        || role_proof.evaluation_time_unix_s() > time_latest
    {
        return Err(EvidenceSourceFreshnessAuthenticationError::RoleProofOutsideTrustedTime);
    }
    if statement.observed_through_unix_ms
        > role_proof.evaluation_time_unix_s().saturating_mul(1000)
    {
        return Err(EvidenceSourceFreshnessAuthenticationError::ObservedThroughAfterAttestation);
    }
    let index_snapshot = bridge_digest(&receipt.index_snapshot_sha256);
    if trusted_time.bindings().iter().any(|binding| {
        binding.source_artifact_sha256() == &index_snapshot
            || binding.source_artifact_sha256() == &retrieval_artifact
    }) {
        return Err(EvidenceSourceFreshnessAuthenticationError::TimeDependsOnSourceSnapshot);
    }

    let source_freshness_sha256 = authenticated_source_freshness_digest(
        frozen_protocol.protocol_sha256(),
        statement,
        freshness_authority,
        role_proof,
    );
    Ok(AuthenticatedEvidenceSourceFreshness {
        source_id: source.source_id.clone(),
        coverage_protocol_sha256: frozen_protocol.protocol_sha256().clone(),
        index_snapshot_sha256: receipt.index_snapshot_sha256.clone(),
        retrieval_artifact_sha256: receipt.retrieval_artifact_sha256.clone(),
        observed_through_unix_ms: statement.observed_through_unix_ms,
        root_authority_sha256: freshness_authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: freshness_authority.trust_snapshot_authority_sha256().clone(),
        freshness_role_authority_sha256: freshness_authority.authority_sha256().clone(),
        role_quorum_proof_sha256: role_proof.proof_sha256().clone(),
        source_freshness_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceFreshnessPolicy {
    pub maximum_search_lag_ms: u64,
    pub maximum_sources: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceFreshnessPolicyError {
    ZeroMaximumSources,
    MaximumSourcesTooLarge,
}

impl EvidenceFreshnessPolicy {
    pub fn validate(&self) -> Result<(), EvidenceFreshnessPolicyError> {
        if self.maximum_sources == 0 {
            return Err(EvidenceFreshnessPolicyError::ZeroMaximumSources);
        }
        if self.maximum_sources > MAX_FRESHNESS_SOURCES {
            return Err(EvidenceFreshnessPolicyError::MaximumSourcesTooLarge);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EvidenceFreshnessClosure {
    FreshWithinFrozenProtocol,
    EvidenceChangedRequiresReview,
    Stale,
    Incomplete,
    Blocked,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum EvidenceFreshnessFinding {
    InvalidPolicy,
    QualifiedClaimMismatch,
    BaselineRelationBindingMismatch,
    BaselineCoverageMismatch,
    BaselineDecisionBindingMismatch,
    BaselineProtocolMismatch,
    BaselineDecisionDoesNotBindCoverage,
    BaselineDecisionNotBound,
    BaselineProtocolLocalCoverageAbsent,
    RefreshClaimBindingMismatch,
    RefreshSubjectMismatch,
    RefreshIntentNotSystematic,
    RefreshDoesNotSupersedeBaseline,
    RefreshCutoffNotAdvanced,
    RefreshProtocolFrozenAfterCutoff,
    DecisionPolicyChanged,
    DeduplicationPolicyChanged,
    SourceSetChanged,
    SourceSemanticsChanged { source_id: ResearchId },
    TooManySources,
    TooManyFreshnessAuthorities,
    RefreshCutoffAfterEvaluation,
    RefreshCutoffTooOld,
    DuplicateFreshnessAuthority { source_id: ResearchId },
    UnknownFreshnessAuthority { source_id: ResearchId },
    MissingFreshnessAuthority { source_id: ResearchId },
    FreshnessProtocolMismatch { source_id: ResearchId },
    FreshnessRootMismatch { source_id: ResearchId },
    FreshnessIndexSnapshotMismatch { source_id: ResearchId },
    FreshnessRetrievalArtifactMismatch { source_id: ResearchId },
    FreshnessDoesNotReachCutoff { source_id: ResearchId },
    RefreshCoverageProtocolMismatch,
    RefreshCoverageReplayMismatch,
    RefreshCoverageIncomplete,
    RefreshCoverageInvalid,
    RefreshRetrievalAccountingAbsent,
    RefreshDecisionCoverageMismatch,
    RefreshDecisionIncomplete,
    RefreshDecisionRejected,
    RefreshDecisionInvalid,
    RefreshProtocolLocalCoverageAbsent,
    RelevantEvidenceOmittedFromClaim { evidence_id: ResearchId },
    ClaimEvidenceNoLongerCovered { evidence_id: ResearchId },
}

pub struct EvidenceRefreshInputs<'a> {
    pub known_evidence: &'a [EvidenceRecord],
    pub receipts: &'a [SourceRetrievalReceipt],
    pub decisions: &'a [CorpusItemDecision],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceFreshnessAssessment {
    qualification_sha256: Sha256Digest,
    baseline_protocol_sha256: Sha256Digest,
    refresh_protocol_sha256: Sha256Digest,
    refresh_coverage_report_sha256: Sha256Digest,
    refresh_decision_binding_sha256: Sha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    refresh_search_cutoff_unix_ms: u64,
    source_freshness_sha256s: Vec<TrustSha256Digest>,
    findings: Vec<EvidenceFreshnessFinding>,
    closure: EvidenceFreshnessClosure,
    assessment_sha256: TrustSha256Digest,
}

impl EvidenceFreshnessAssessment {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn baseline_protocol_sha256(&self) -> &Sha256Digest { &self.baseline_protocol_sha256 }
    pub fn refresh_protocol_sha256(&self) -> &Sha256Digest { &self.refresh_protocol_sha256 }
    pub fn refresh_coverage_report_sha256(&self) -> &Sha256Digest {
        &self.refresh_coverage_report_sha256
    }
    pub fn refresh_decision_binding_sha256(&self) -> &Sha256Digest {
        &self.refresh_decision_binding_sha256
    }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn refresh_search_cutoff_unix_ms(&self) -> u64 { self.refresh_search_cutoff_unix_ms }
    pub fn source_freshness_sha256s(&self) -> &[TrustSha256Digest] {
        &self.source_freshness_sha256s
    }
    pub fn findings(&self) -> &[EvidenceFreshnessFinding] { &self.findings }
    pub fn closure(&self) -> EvidenceFreshnessClosure { self.closure }
    pub fn assessment_sha256(&self) -> &TrustSha256Digest { &self.assessment_sha256 }
    pub fn evidence_freshness_within_frozen_protocol_established(&self) -> bool {
        self.closure == EvidenceFreshnessClosure::FreshWithinFrozenProtocol
    }
    pub const fn global_evidence_exhaustiveness_established(&self) -> bool { false }
    pub const fn current_validity_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

#[allow(clippy::too_many_arguments)]
pub fn assess_evidence_freshness(
    qualified: &QualifiedScientificClaim,
    claim: &ScientificClaim,
    claim_binding: &ClaimRelationBindingReport,
    baseline_protocol: &FrozenEvidenceCoverageProtocol,
    baseline_coverage: &EvidenceCoverageAuthorityReport,
    baseline_decision_binding: &EvidenceDecisionBindingReport,
    refresh_protocol: &FrozenEvidenceCoverageProtocol,
    refresh_coverage: &EvidenceCoverageAuthorityReport,
    refresh_decision_binding: &EvidenceDecisionBindingReport,
    refresh_inputs: EvidenceRefreshInputs<'_>,
    source_freshness: &[AuthenticatedEvidenceSourceFreshness],
    evaluation_time: &TrustedTime,
    policy: &EvidenceFreshnessPolicy,
) -> EvidenceFreshnessAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut blocked = false;
    let mut stale = false;
    let mut evidence_changed = false;

    if policy.validate().is_err() {
        findings.push(EvidenceFreshnessFinding::InvalidPolicy);
        invalid = true;
    }
    if qualified.claim_id() != claim.claim_id() || qualified.subject_sha256() != claim.subject_sha256() {
        findings.push(EvidenceFreshnessFinding::QualifiedClaimMismatch);
        invalid = true;
    }
    let profile = qualified.decision().authorized_profile().profile().profile();
    if &profile.relation_binding_sha256 != claim_binding.binding_sha256() {
        findings.push(EvidenceFreshnessFinding::BaselineRelationBindingMismatch);
        invalid = true;
    }
    if &profile.evidence_coverage_report_sha256 != baseline_coverage.report_sha256() {
        findings.push(EvidenceFreshnessFinding::BaselineCoverageMismatch);
        invalid = true;
    }
    if &profile.evidence_decision_binding_sha256 != baseline_decision_binding.report_sha256() {
        findings.push(EvidenceFreshnessFinding::BaselineDecisionBindingMismatch);
        invalid = true;
    }
    if baseline_coverage.protocol_sha256() != baseline_protocol.protocol_sha256() {
        findings.push(EvidenceFreshnessFinding::BaselineProtocolMismatch);
        invalid = true;
    }
    if baseline_decision_binding.source_coverage_report_sha256() != baseline_coverage.report_sha256() {
        findings.push(EvidenceFreshnessFinding::BaselineDecisionDoesNotBindCoverage);
        invalid = true;
    }
    if baseline_decision_binding.closure() != EvidenceDecisionBindingClosure::Bound {
        findings.push(EvidenceFreshnessFinding::BaselineDecisionNotBound);
        invalid = true;
    }
    if !baseline_decision_binding.protocol_local_coverage_established() {
        findings.push(EvidenceFreshnessFinding::BaselineProtocolLocalCoverageAbsent);
        invalid = true;
    }

    let baseline = baseline_protocol.protocol();
    let refresh = refresh_protocol.protocol();
    if &refresh.claim_binding_sha256 != claim_binding.binding_sha256() {
        findings.push(EvidenceFreshnessFinding::RefreshClaimBindingMismatch);
        invalid = true;
    }
    if &refresh.subject_sha256 != claim.subject_sha256() {
        findings.push(EvidenceFreshnessFinding::RefreshSubjectMismatch);
        invalid = true;
    }
    if refresh.intent != crate::EvidenceCoverageIntent::SystematicClaimAudit {
        findings.push(EvidenceFreshnessFinding::RefreshIntentNotSystematic);
        invalid = true;
    }
    if refresh.supersedes_protocol_sha256.as_ref() != Some(baseline_protocol.protocol_sha256()) {
        findings.push(EvidenceFreshnessFinding::RefreshDoesNotSupersedeBaseline);
        invalid = true;
    }
    if refresh.declared_search_cutoff_unix_ms <= baseline.declared_search_cutoff_unix_ms {
        findings.push(EvidenceFreshnessFinding::RefreshCutoffNotAdvanced);
        invalid = true;
    }
    if refresh.declared_frozen_at_unix_ms > refresh.declared_search_cutoff_unix_ms {
        findings.push(EvidenceFreshnessFinding::RefreshProtocolFrozenAfterCutoff);
        invalid = true;
    }
    if refresh.decision_policy_sha256 != baseline.decision_policy_sha256 {
        findings.push(EvidenceFreshnessFinding::DecisionPolicyChanged);
        invalid = true;
    }
    if refresh.deduplication_policy_sha256 != baseline.deduplication_policy_sha256 {
        findings.push(EvidenceFreshnessFinding::DeduplicationPolicyChanged);
        invalid = true;
    }

    let baseline_sources: BTreeMap<_, _> = baseline
        .sources
        .iter()
        .map(|source| (source.source_id.clone(), source))
        .collect();
    let refresh_sources: BTreeMap<_, _> = refresh
        .sources
        .iter()
        .map(|source| (source.source_id.clone(), source))
        .collect();
    if baseline_sources.keys().collect::<BTreeSet<_>>()
        != refresh_sources.keys().collect::<BTreeSet<_>>()
    {
        findings.push(EvidenceFreshnessFinding::SourceSetChanged);
        invalid = true;
    }
    for (source_id, baseline_source) in &baseline_sources {
        if let Some(refresh_source) = refresh_sources.get(source_id) {
            if baseline_source.source_class != refresh_source.source_class
                || baseline_source.source_scope_sha256 != refresh_source.source_scope_sha256
                || baseline_source.query_sha256 != refresh_source.query_sha256
                || baseline_source.retrieval_protocol_sha256 != refresh_source.retrieval_protocol_sha256
            {
                findings.push(EvidenceFreshnessFinding::SourceSemanticsChanged {
                    source_id: source_id.clone(),
                });
                invalid = true;
            }
        }
    }
    if refresh.sources.len() > policy.maximum_sources || refresh.sources.len() > MAX_FRESHNESS_SOURCES {
        findings.push(EvidenceFreshnessFinding::TooManySources);
        invalid = true;
    }
    if source_freshness.len() > policy.maximum_sources
        || source_freshness.len() > MAX_FRESHNESS_SOURCES
    {
        findings.push(EvidenceFreshnessFinding::TooManyFreshnessAuthorities);
        invalid = true;
    }

    let (evaluation_earliest_s, evaluation_latest_s) = evaluation_time.consensus_interval();
    let evaluation_earliest_ms = evaluation_earliest_s.saturating_mul(1000);
    let evaluation_latest_ms = evaluation_latest_s.saturating_mul(1000);
    if refresh.declared_search_cutoff_unix_ms > evaluation_earliest_ms {
        findings.push(EvidenceFreshnessFinding::RefreshCutoffAfterEvaluation);
        blocked = true;
    }
    if evaluation_latest_ms
        > refresh
            .declared_search_cutoff_unix_ms
            .saturating_add(policy.maximum_search_lag_ms)
    {
        findings.push(EvidenceFreshnessFinding::RefreshCutoffTooOld);
        stale = true;
    }

    let receipts_by_source: BTreeMap<_, _> = refresh_inputs
        .receipts
        .iter()
        .map(|receipt| (receipt.source_id.clone(), receipt))
        .collect();
    let mut freshness_by_source = BTreeMap::new();
    for item in source_freshness {
        if freshness_by_source.insert(item.source_id().clone(), item).is_some() {
            findings.push(EvidenceFreshnessFinding::DuplicateFreshnessAuthority {
                source_id: item.source_id().clone(),
            });
            invalid = true;
        }
        let Some(source) = refresh_sources.get(item.source_id()) else {
            findings.push(EvidenceFreshnessFinding::UnknownFreshnessAuthority {
                source_id: item.source_id().clone(),
            });
            invalid = true;
            continue;
        };
        let Some(receipt) = receipts_by_source.get(item.source_id()).copied() else {
            findings.push(EvidenceFreshnessFinding::UnknownFreshnessAuthority {
                source_id: item.source_id().clone(),
            });
            invalid = true;
            continue;
        };
        if item.coverage_protocol_sha256() != refresh_protocol.protocol_sha256() {
            findings.push(EvidenceFreshnessFinding::FreshnessProtocolMismatch {
                source_id: source.source_id.clone(),
            });
            invalid = true;
        }
        if item.root_authority_sha256() != evaluation_time.root_authority_sha256() {
            findings.push(EvidenceFreshnessFinding::FreshnessRootMismatch {
                source_id: source.source_id.clone(),
            });
            invalid = true;
        }
        if item.index_snapshot_sha256() != &receipt.index_snapshot_sha256 {
            findings.push(EvidenceFreshnessFinding::FreshnessIndexSnapshotMismatch {
                source_id: source.source_id.clone(),
            });
            invalid = true;
        }
        if item.retrieval_artifact_sha256() != &receipt.retrieval_artifact_sha256 {
            findings.push(EvidenceFreshnessFinding::FreshnessRetrievalArtifactMismatch {
                source_id: source.source_id.clone(),
            });
            invalid = true;
        }
        if item.observed_through_unix_ms() < refresh.declared_search_cutoff_unix_ms {
            findings.push(EvidenceFreshnessFinding::FreshnessDoesNotReachCutoff {
                source_id: source.source_id.clone(),
            });
            incomplete = true;
        }
    }
    for source in &refresh.sources {
        if !freshness_by_source.contains_key(&source.source_id) {
            findings.push(EvidenceFreshnessFinding::MissingFreshnessAuthority {
                source_id: source.source_id.clone(),
            });
            incomplete = true;
        }
    }

    if refresh_coverage.protocol_sha256() != refresh_protocol.protocol_sha256() {
        findings.push(EvidenceFreshnessFinding::RefreshCoverageProtocolMismatch);
        invalid = true;
    }
    let replayed = audit_claim_evidence_coverage(
        claim,
        claim_binding,
        refresh_protocol,
        refresh_inputs.known_evidence,
        refresh_inputs.receipts,
        refresh_inputs.decisions,
    );
    if replayed.report_sha256() != refresh_coverage.report_sha256() {
        findings.push(EvidenceFreshnessFinding::RefreshCoverageReplayMismatch);
        invalid = true;
    }
    match refresh_coverage.closure() {
        EvidenceCoverageClosure::CompleteWithinProtocol => {}
        EvidenceCoverageClosure::Incomplete => {
            findings.push(EvidenceFreshnessFinding::RefreshCoverageIncomplete);
            incomplete = true;
        }
        EvidenceCoverageClosure::Invalid => {
            findings.push(EvidenceFreshnessFinding::RefreshCoverageInvalid);
            invalid = true;
        }
    }
    if !refresh_coverage.retrieval_accounting_established() {
        findings.push(EvidenceFreshnessFinding::RefreshRetrievalAccountingAbsent);
        incomplete = true;
    }
    if refresh_decision_binding.source_coverage_report_sha256() != refresh_coverage.report_sha256() {
        findings.push(EvidenceFreshnessFinding::RefreshDecisionCoverageMismatch);
        invalid = true;
    }
    match refresh_decision_binding.closure() {
        EvidenceDecisionBindingClosure::Bound => {}
        EvidenceDecisionBindingClosure::Incomplete => {
            findings.push(EvidenceFreshnessFinding::RefreshDecisionIncomplete);
            incomplete = true;
        }
        EvidenceDecisionBindingClosure::Rejected => {
            findings.push(EvidenceFreshnessFinding::RefreshDecisionRejected);
            blocked = true;
        }
        EvidenceDecisionBindingClosure::Invalid => {
            findings.push(EvidenceFreshnessFinding::RefreshDecisionInvalid);
            invalid = true;
        }
    }
    if !refresh_decision_binding.protocol_local_coverage_established() {
        findings.push(EvidenceFreshnessFinding::RefreshProtocolLocalCoverageAbsent);
        incomplete = true;
    }

    for evidence_id in refresh_coverage.relevant_evidence_omitted_from_claim() {
        findings.push(EvidenceFreshnessFinding::RelevantEvidenceOmittedFromClaim {
            evidence_id: evidence_id.clone(),
        });
        evidence_changed = true;
    }
    for evidence_id in refresh_coverage.claim_evidence_not_covered() {
        findings.push(EvidenceFreshnessFinding::ClaimEvidenceNoLongerCovered {
            evidence_id: evidence_id.clone(),
        });
        evidence_changed = true;
    }

    findings.sort();
    findings.dedup();
    let closure = if invalid {
        EvidenceFreshnessClosure::Invalid
    } else if evidence_changed {
        EvidenceFreshnessClosure::EvidenceChangedRequiresReview
    } else if blocked {
        EvidenceFreshnessClosure::Blocked
    } else if stale {
        EvidenceFreshnessClosure::Stale
    } else if incomplete {
        EvidenceFreshnessClosure::Incomplete
    } else {
        EvidenceFreshnessClosure::FreshWithinFrozenProtocol
    };

    let mut source_freshness_sha256s: Vec<_> = source_freshness
        .iter()
        .map(|item| item.source_freshness_sha256().clone())
        .collect();
    source_freshness_sha256s.sort();
    let assessment_sha256 = evidence_freshness_assessment_digest(
        qualified,
        baseline_protocol,
        refresh_protocol,
        refresh_coverage,
        refresh_decision_binding,
        evaluation_time,
        &source_freshness_sha256s,
        policy,
        &findings,
        closure,
    );
    EvidenceFreshnessAssessment {
        qualification_sha256: qualified.qualification_sha256().clone(),
        baseline_protocol_sha256: baseline_protocol.protocol_sha256().clone(),
        refresh_protocol_sha256: refresh_protocol.protocol_sha256().clone(),
        refresh_coverage_report_sha256: refresh_coverage.report_sha256().clone(),
        refresh_decision_binding_sha256: refresh_decision_binding.report_sha256().clone(),
        evaluation_time_authority_sha256: evaluation_time.authority_sha256().clone(),
        refresh_search_cutoff_unix_ms: refresh.declared_search_cutoff_unix_ms,
        source_freshness_sha256s,
        findings,
        closure,
        assessment_sha256,
    }
}

fn source_freshness_statement_digest(
    source: &EvidenceSourceSpec,
    receipt: &SourceRetrievalReceipt,
    observed_through_unix_ms: u64,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(EVIDENCE_SOURCE_FRESHNESS_STATEMENT_DOMAIN);
    digest.text(source.source_id.as_str());
    digest.text(source.source_scope_sha256.as_str());
    digest.text(source.query_sha256.as_str());
    digest.text(source.retrieval_protocol_sha256.as_str());
    digest.text(receipt.index_snapshot_sha256.as_str());
    digest.text(receipt.retrieval_artifact_sha256.as_str());
    digest.text(&observed_through_unix_ms.to_string());
    digest.digest()
}

fn authenticated_source_freshness_digest(
    protocol_sha256: &Sha256Digest,
    statement: &EvidenceSourceFreshnessStatement,
    authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(AUTHENTICATED_EVIDENCE_SOURCE_FRESHNESS_DOMAIN);
    digest.text(protocol_sha256.as_str());
    digest.text(statement.statement_sha256().as_str());
    digest.text(authority.root_authority_sha256().as_str());
    digest.text(authority.trust_snapshot_authority_sha256().as_str());
    digest.text(authority.authority_sha256().as_str());
    digest.text(role_proof.proof_sha256().as_str());
    digest.text(&role_proof.evaluation_time_unix_s().to_string());
    digest.text("source-freshness-established");
    digest.text("global-source-exhaustiveness-not-established");
    digest.digest()
}

#[allow(clippy::too_many_arguments)]
fn evidence_freshness_assessment_digest(
    qualified: &QualifiedScientificClaim,
    baseline_protocol: &FrozenEvidenceCoverageProtocol,
    refresh_protocol: &FrozenEvidenceCoverageProtocol,
    refresh_coverage: &EvidenceCoverageAuthorityReport,
    refresh_decision_binding: &EvidenceDecisionBindingReport,
    evaluation_time: &TrustedTime,
    source_freshness_sha256s: &[TrustSha256Digest],
    policy: &EvidenceFreshnessPolicy,
    findings: &[EvidenceFreshnessFinding],
    closure: EvidenceFreshnessClosure,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(EVIDENCE_FRESHNESS_ASSESSMENT_DOMAIN);
    digest.text(qualified.qualification_sha256().as_str());
    digest.text(baseline_protocol.protocol_sha256().as_str());
    digest.text(refresh_protocol.protocol_sha256().as_str());
    digest.text(refresh_coverage.report_sha256().as_str());
    digest.text(refresh_decision_binding.report_sha256().as_str());
    digest.text(evaluation_time.authority_sha256().as_str());
    for value in source_freshness_sha256s { digest.text(value.as_str()); }
    digest.text(&policy.maximum_search_lag_ms.to_string());
    digest.text(&policy.maximum_sources.to_string());
    for finding in findings { digest_finding(&mut digest, finding); }
    digest.text(match closure {
        EvidenceFreshnessClosure::FreshWithinFrozenProtocol => "fresh-within-frozen-protocol",
        EvidenceFreshnessClosure::EvidenceChangedRequiresReview => "evidence-changed-requires-review",
        EvidenceFreshnessClosure::Stale => "stale",
        EvidenceFreshnessClosure::Incomplete => "incomplete",
        EvidenceFreshnessClosure::Blocked => "blocked",
        EvidenceFreshnessClosure::Invalid => "invalid",
    });
    digest.text("global-evidence-exhaustiveness-not-established");
    digest.text("current-validity-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn digest_finding(digest: &mut FramedDigest, finding: &EvidenceFreshnessFinding) {
    use EvidenceFreshnessFinding as Finding;
    match finding {
        Finding::InvalidPolicy => digest.text("invalid-policy"),
        Finding::QualifiedClaimMismatch => digest.text("qualified-claim-mismatch"),
        Finding::BaselineRelationBindingMismatch => digest.text("baseline-relation-binding-mismatch"),
        Finding::BaselineCoverageMismatch => digest.text("baseline-coverage-mismatch"),
        Finding::BaselineDecisionBindingMismatch => digest.text("baseline-decision-binding-mismatch"),
        Finding::BaselineProtocolMismatch => digest.text("baseline-protocol-mismatch"),
        Finding::BaselineDecisionDoesNotBindCoverage => digest.text("baseline-decision-does-not-bind-coverage"),
        Finding::BaselineDecisionNotBound => digest.text("baseline-decision-not-bound"),
        Finding::BaselineProtocolLocalCoverageAbsent => digest.text("baseline-protocol-local-coverage-absent"),
        Finding::RefreshClaimBindingMismatch => digest.text("refresh-claim-binding-mismatch"),
        Finding::RefreshSubjectMismatch => digest.text("refresh-subject-mismatch"),
        Finding::RefreshIntentNotSystematic => digest.text("refresh-intent-not-systematic"),
        Finding::RefreshDoesNotSupersedeBaseline => digest.text("refresh-does-not-supersede-baseline"),
        Finding::RefreshCutoffNotAdvanced => digest.text("refresh-cutoff-not-advanced"),
        Finding::RefreshProtocolFrozenAfterCutoff => digest.text("refresh-protocol-frozen-after-cutoff"),
        Finding::DecisionPolicyChanged => digest.text("decision-policy-changed"),
        Finding::DeduplicationPolicyChanged => digest.text("deduplication-policy-changed"),
        Finding::SourceSetChanged => digest.text("source-set-changed"),
        Finding::SourceSemanticsChanged { source_id } => {
            digest.text("source-semantics-changed");
            digest.text(source_id.as_str());
        }
        Finding::TooManySources => digest.text("too-many-sources"),
        Finding::TooManyFreshnessAuthorities => digest.text("too-many-freshness-authorities"),
        Finding::RefreshCutoffAfterEvaluation => digest.text("refresh-cutoff-after-evaluation"),
        Finding::RefreshCutoffTooOld => digest.text("refresh-cutoff-too-old"),
        Finding::DuplicateFreshnessAuthority { source_id } => {
            digest.text("duplicate-freshness-authority");
            digest.text(source_id.as_str());
        }
        Finding::UnknownFreshnessAuthority { source_id } => {
            digest.text("unknown-freshness-authority");
            digest.text(source_id.as_str());
        }
        Finding::MissingFreshnessAuthority { source_id } => {
            digest.text("missing-freshness-authority");
            digest.text(source_id.as_str());
        }
        Finding::FreshnessProtocolMismatch { source_id } => {
            digest.text("freshness-protocol-mismatch");
            digest.text(source_id.as_str());
        }
        Finding::FreshnessRootMismatch { source_id } => {
            digest.text("freshness-root-mismatch");
            digest.text(source_id.as_str());
        }
        Finding::FreshnessIndexSnapshotMismatch { source_id } => {
            digest.text("freshness-index-snapshot-mismatch");
            digest.text(source_id.as_str());
        }
        Finding::FreshnessRetrievalArtifactMismatch { source_id } => {
            digest.text("freshness-retrieval-artifact-mismatch");
            digest.text(source_id.as_str());
        }
        Finding::FreshnessDoesNotReachCutoff { source_id } => {
            digest.text("freshness-does-not-reach-cutoff");
            digest.text(source_id.as_str());
        }
        Finding::RefreshCoverageProtocolMismatch => digest.text("refresh-coverage-protocol-mismatch"),
        Finding::RefreshCoverageReplayMismatch => digest.text("refresh-coverage-replay-mismatch"),
        Finding::RefreshCoverageIncomplete => digest.text("refresh-coverage-incomplete"),
        Finding::RefreshCoverageInvalid => digest.text("refresh-coverage-invalid"),
        Finding::RefreshRetrievalAccountingAbsent => digest.text("refresh-retrieval-accounting-absent"),
        Finding::RefreshDecisionCoverageMismatch => digest.text("refresh-decision-coverage-mismatch"),
        Finding::RefreshDecisionIncomplete => digest.text("refresh-decision-incomplete"),
        Finding::RefreshDecisionRejected => digest.text("refresh-decision-rejected"),
        Finding::RefreshDecisionInvalid => digest.text("refresh-decision-invalid"),
        Finding::RefreshProtocolLocalCoverageAbsent => digest.text("refresh-protocol-local-coverage-absent"),
        Finding::RelevantEvidenceOmittedFromClaim { evidence_id } => {
            digest.text("relevant-evidence-omitted-from-claim");
            digest.text(evidence_id.as_str());
        }
        Finding::ClaimEvidenceNoLongerCovered { evidence_id } => {
            digest.text("claim-evidence-no-longer-covered");
            digest.text(evidence_id.as_str());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_search_lag_is_valid_but_source_capacity_must_be_nonzero() {
        assert!(EvidenceFreshnessPolicy {
            maximum_search_lag_ms: 0,
            maximum_sources: 1,
        }
        .validate()
        .is_ok());
        assert_eq!(
            EvidenceFreshnessPolicy {
                maximum_search_lag_ms: 1,
                maximum_sources: 0,
            }
            .validate(),
            Err(EvidenceFreshnessPolicyError::ZeroMaximumSources)
        );
    }
}
