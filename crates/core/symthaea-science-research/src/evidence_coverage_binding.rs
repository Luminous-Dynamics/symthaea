// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Semantic binding for evidence-corpus inclusion/exclusion decisions.
//!
//! SCI-011A can bind exact retrieval accounting, but corpus relevance decisions
//! remain caller-declared. This layer reviews the exact include/exclude/duplicate
//! classifications under a frozen policy. It may establish protocol-local
//! coverage only when retrieval accounting was already complete, every reviewable
//! decision is accepted, no item remains unresolved, and the protocol intent was
//! systematic. Global exhaustiveness and scientific qualification remain false.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    AuthorityLevel, ClaimRelationBindingReport, CorpusDecisionKind, CorpusItemDecision,
    EvidenceCoverageAuthorityReport, EvidenceCoverageClosure, EvidenceCoverageIntent,
    EvidenceRecord, FrozenEvidenceCoverageProtocol, ResearchId, ScientificClaim,
    SemanticReviewOutcome, Sha256Digest, SourceRetrievalReceipt,
    audit_claim_evidence_coverage,
};

pub const EVIDENCE_DECISION_BINDING_SCHEMA: &str =
    "symthaea.evidence-coverage-decision-binding.v1";
const DECISION_BINDING_POLICY_DOMAIN: &str =
    "symthaea.evidence-coverage-decision-binding-policy.identity.v1";
const DECISION_BINDING_REPORT_DOMAIN: &str =
    "symthaea.evidence-coverage-decision-binding-report.identity.v1";
const DECISION_DIGEST_DOMAIN: &str = "symthaea.evidence-corpus-decision.identity.v1";
const DECISION_REVIEW_SET_DOMAIN: &str =
    "symthaea.evidence-coverage-decision-review-set.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum CorpusDecisionClass {
    Include,
    Exclude,
    Duplicate,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum CorpusDecisionReviewDimension {
    ClaimRelevance,
    InclusionEligibility,
    ExclusionJustification,
    DuplicateIdentity,
    ScopeCompatibility,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusDecisionReviewRequirement {
    pub requirement_id: ResearchId,
    pub decision_class: CorpusDecisionClass,
    pub dimension: CorpusDecisionReviewDimension,
    pub reviewer_identity_sha256: Sha256Digest,
    pub review_protocol_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDecisionBindingPolicy {
    pub schema_version: String,
    pub policy_id: ResearchId,
    pub coverage_report_sha256: Sha256Digest,
    pub requirements: Vec<CorpusDecisionReviewRequirement>,
    pub supersedes_policy_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceDecisionBindingPolicyIssue {
    WrongSchemaVersion { found: String },
    MissingRequirements,
    DuplicateRequirementId { requirement_id: ResearchId },
}

impl EvidenceDecisionBindingPolicy {
    pub fn validate(&self) -> Vec<EvidenceDecisionBindingPolicyIssue> {
        let mut issues = Vec::new();
        if self.schema_version != EVIDENCE_DECISION_BINDING_SCHEMA {
            issues.push(EvidenceDecisionBindingPolicyIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.requirements.is_empty() {
            issues.push(EvidenceDecisionBindingPolicyIssue::MissingRequirements);
        }
        let mut ids = BTreeSet::new();
        for requirement in &self.requirements {
            if !ids.insert(requirement.requirement_id.clone()) {
                issues.push(EvidenceDecisionBindingPolicyIssue::DuplicateRequirementId {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }
        }
        issues
    }

    pub fn freeze(
        self,
    ) -> Result<FrozenEvidenceDecisionBindingPolicy, Vec<EvidenceDecisionBindingPolicyIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let policy_sha256 = policy_digest(&self);
        Ok(FrozenEvidenceDecisionBindingPolicy {
            policy: self,
            policy_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenEvidenceDecisionBindingPolicy {
    policy: EvidenceDecisionBindingPolicy,
    policy_sha256: Sha256Digest,
}

impl FrozenEvidenceDecisionBindingPolicy {
    pub fn policy(&self) -> &EvidenceDecisionBindingPolicy {
        &self.policy
    }

    pub fn policy_sha256(&self) -> &Sha256Digest {
        &self.policy_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusDecisionSemanticReview {
    pub requirement_id: ResearchId,
    pub source_id: ResearchId,
    pub item_sha256: Sha256Digest,
    pub decision_sha256: Sha256Digest,
    pub decision_class: CorpusDecisionClass,
    pub dimension: CorpusDecisionReviewDimension,
    pub reviewer_identity_sha256: Sha256Digest,
    pub review_protocol_sha256: Sha256Digest,
    pub review_artifact_sha256: Sha256Digest,
    pub outcome: SemanticReviewOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceDecisionBindingIssue {
    CoverageAuditCouldNotBeReplayed,
    CoverageReportMismatch,
    PolicyCoverageReportMismatch,
    MissingRequirementForDecisionClass { decision_class: CorpusDecisionClass },
    DuplicateDecisionReview {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
        requirement_id: ResearchId,
    },
    UnknownRequirement { requirement_id: ResearchId },
    UnknownDecision {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    DecisionClassMismatch {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    DecisionIdentityMismatch {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    ReviewDimensionMismatch {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    ReviewerIdentityMismatch {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
    ReviewProtocolMismatch {
        source_id: ResearchId,
        item_sha256: Sha256Digest,
    },
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceDecisionBindingClosure {
    Bound,
    Incomplete,
    Rejected,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceDecisionBindingReport {
    report_sha256: Sha256Digest,
    source_coverage_report_sha256: Sha256Digest,
    policy_sha256: Sha256Digest,
    review_set_sha256: Sha256Digest,
    closure: EvidenceDecisionBindingClosure,
    decision_binding_level: AuthorityLevel,
    claim_coverage_level: AuthorityLevel,
    protocol_local_coverage_established: bool,
    global_exhaustiveness_established: bool,
    missing_review_keys: BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
    rejected_review_keys: BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
    inconclusive_review_keys: BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
    invalid_review_keys: BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
    qualification_established: bool,
}

impl EvidenceDecisionBindingReport {
    pub fn report_sha256(&self) -> &Sha256Digest {
        &self.report_sha256
    }

    pub fn source_coverage_report_sha256(&self) -> &Sha256Digest {
        &self.source_coverage_report_sha256
    }

    pub fn policy_sha256(&self) -> &Sha256Digest {
        &self.policy_sha256
    }

    pub fn review_set_sha256(&self) -> &Sha256Digest {
        &self.review_set_sha256
    }

    pub fn closure(&self) -> EvidenceDecisionBindingClosure {
        self.closure
    }

    pub fn decision_binding_level(&self) -> AuthorityLevel {
        self.decision_binding_level
    }

    pub fn claim_coverage_level(&self) -> AuthorityLevel {
        self.claim_coverage_level
    }

    pub fn protocol_local_coverage_established(&self) -> bool {
        self.protocol_local_coverage_established
    }

    pub fn global_exhaustiveness_established(&self) -> bool {
        self.global_exhaustiveness_established
    }

    pub fn qualification_established(&self) -> bool {
        self.qualification_established
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bind_evidence_coverage_decisions(
    claim: &ScientificClaim,
    claim_binding: &ClaimRelationBindingReport,
    frozen_coverage_protocol: &FrozenEvidenceCoverageProtocol,
    known_evidence: &[EvidenceRecord],
    receipts: &[SourceRetrievalReceipt],
    decisions: &[CorpusItemDecision],
    declared_coverage: &EvidenceCoverageAuthorityReport,
    frozen_policy: &FrozenEvidenceDecisionBindingPolicy,
    reviews: &[CorpusDecisionSemanticReview],
) -> Result<EvidenceDecisionBindingReport, Vec<EvidenceDecisionBindingIssue>> {
    let replayed = audit_claim_evidence_coverage(
        claim,
        claim_binding,
        frozen_coverage_protocol,
        known_evidence,
        receipts,
        decisions,
    );
    if replayed.report_sha256() != declared_coverage.report_sha256() {
        return Err(vec![EvidenceDecisionBindingIssue::CoverageReportMismatch]);
    }

    let policy = frozen_policy.policy();
    if &policy.coverage_report_sha256 != declared_coverage.report_sha256() {
        return Err(vec![EvidenceDecisionBindingIssue::PolicyCoverageReportMismatch]);
    }

    let mut decisions_by_key = BTreeMap::new();
    for decision in decisions {
        decisions_by_key.insert(
            (decision.source_id.clone(), decision.item_sha256.clone()),
            decision,
        );
    }
    let requirements_by_id: BTreeMap<_, _> = policy
        .requirements
        .iter()
        .map(|requirement| (requirement.requirement_id.clone(), requirement))
        .collect();

    let present_classes: BTreeSet<_> = decisions
        .iter()
        .filter_map(|decision| decision_class(&decision.decision))
        .collect();
    let policy_classes: BTreeSet<_> = policy
        .requirements
        .iter()
        .map(|requirement| requirement.decision_class)
        .collect();

    let mut issues = Vec::new();
    for decision_class in &present_classes {
        if !policy_classes.contains(decision_class) {
            issues.push(EvidenceDecisionBindingIssue::MissingRequirementForDecisionClass {
                decision_class: *decision_class,
            });
        }
    }

    let mut reviews_by_key = BTreeMap::new();
    for review in reviews {
        let key = (
            review.source_id.clone(),
            review.item_sha256.clone(),
            review.requirement_id.clone(),
        );
        if reviews_by_key.insert(key.clone(), review).is_some() {
            issues.push(EvidenceDecisionBindingIssue::DuplicateDecisionReview {
                source_id: key.0,
                item_sha256: key.1,
                requirement_id: key.2,
            });
            continue;
        }
        let Some(requirement) = requirements_by_id.get(&review.requirement_id) else {
            issues.push(EvidenceDecisionBindingIssue::UnknownRequirement {
                requirement_id: review.requirement_id.clone(),
            });
            continue;
        };
        let decision_key = (review.source_id.clone(), review.item_sha256.clone());
        let Some(decision) = decisions_by_key.get(&decision_key) else {
            issues.push(EvidenceDecisionBindingIssue::UnknownDecision {
                source_id: review.source_id.clone(),
                item_sha256: review.item_sha256.clone(),
            });
            continue;
        };
        let Some(actual_class) = decision_class(&decision.decision) else {
            issues.push(EvidenceDecisionBindingIssue::DecisionClassMismatch {
                source_id: review.source_id.clone(),
                item_sha256: review.item_sha256.clone(),
            });
            continue;
        };
        if actual_class != requirement.decision_class || review.decision_class != actual_class {
            issues.push(EvidenceDecisionBindingIssue::DecisionClassMismatch {
                source_id: review.source_id.clone(),
                item_sha256: review.item_sha256.clone(),
            });
        }
        if review.decision_sha256 != corpus_decision_digest(decision) {
            issues.push(EvidenceDecisionBindingIssue::DecisionIdentityMismatch {
                source_id: review.source_id.clone(),
                item_sha256: review.item_sha256.clone(),
            });
        }
        if review.dimension != requirement.dimension {
            issues.push(EvidenceDecisionBindingIssue::ReviewDimensionMismatch {
                source_id: review.source_id.clone(),
                item_sha256: review.item_sha256.clone(),
            });
        }
        if review.reviewer_identity_sha256 != requirement.reviewer_identity_sha256 {
            issues.push(EvidenceDecisionBindingIssue::ReviewerIdentityMismatch {
                source_id: review.source_id.clone(),
                item_sha256: review.item_sha256.clone(),
            });
        }
        if review.review_protocol_sha256 != requirement.review_protocol_sha256 {
            issues.push(EvidenceDecisionBindingIssue::ReviewProtocolMismatch {
                source_id: review.source_id.clone(),
                item_sha256: review.item_sha256.clone(),
            });
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let mut required_review_keys = BTreeSet::new();
    for decision in decisions {
        let Some(class) = decision_class(&decision.decision) else {
            continue;
        };
        for requirement in policy
            .requirements
            .iter()
            .filter(|requirement| requirement.decision_class == class)
        {
            required_review_keys.insert((
                decision.source_id.clone(),
                decision.item_sha256.clone(),
                requirement.requirement_id.clone(),
            ));
        }
    }

    let mut missing_review_keys = BTreeSet::new();
    let mut rejected_review_keys = BTreeSet::new();
    let mut inconclusive_review_keys = BTreeSet::new();
    let mut invalid_review_keys = BTreeSet::new();

    for key in &required_review_keys {
        let Some(review) = reviews_by_key.get(key) else {
            missing_review_keys.insert(key.clone());
            continue;
        };
        match review.outcome {
            SemanticReviewOutcome::Accepted => {}
            SemanticReviewOutcome::Rejected => {
                rejected_review_keys.insert(key.clone());
            }
            SemanticReviewOutcome::Inconclusive => {
                inconclusive_review_keys.insert(key.clone());
            }
            SemanticReviewOutcome::Invalid => {
                invalid_review_keys.insert(key.clone());
            }
        }
    }

    let has_unresolved_decisions = decisions
        .iter()
        .any(|decision| matches!(decision.decision, CorpusDecisionKind::Unresolved));
    let source_invalid = declared_coverage.closure() == EvidenceCoverageClosure::Invalid;
    let source_incomplete = declared_coverage.closure() == EvidenceCoverageClosure::Incomplete;

    let closure = if source_invalid || !invalid_review_keys.is_empty() {
        EvidenceDecisionBindingClosure::Invalid
    } else if !rejected_review_keys.is_empty() {
        EvidenceDecisionBindingClosure::Rejected
    } else if source_incomplete
        || !missing_review_keys.is_empty()
        || !inconclusive_review_keys.is_empty()
        || has_unresolved_decisions
        || required_review_keys.is_empty()
    {
        EvidenceDecisionBindingClosure::Incomplete
    } else {
        EvidenceDecisionBindingClosure::Bound
    };

    let decision_binding_level = match closure {
        EvidenceDecisionBindingClosure::Bound => AuthorityLevel::Bound,
        EvidenceDecisionBindingClosure::Invalid => AuthorityLevel::None,
        EvidenceDecisionBindingClosure::Incomplete | EvidenceDecisionBindingClosure::Rejected => {
            AuthorityLevel::Declared
        }
    };

    let protocol_local_coverage_established = closure == EvidenceDecisionBindingClosure::Bound
        && declared_coverage.retrieval_accounting_established()
        && frozen_coverage_protocol.protocol().intent == EvidenceCoverageIntent::SystematicClaimAudit
        && declared_coverage.relevant_evidence_omitted_from_claim().is_empty()
        && declared_coverage.claim_evidence_not_covered().is_empty()
        && declared_coverage.unresolved_items().is_empty();

    let claim_coverage_level = if closure == EvidenceDecisionBindingClosure::Invalid {
        AuthorityLevel::None
    } else if protocol_local_coverage_established {
        AuthorityLevel::Bound
    } else {
        AuthorityLevel::Declared
    };

    let review_set_sha256 = review_set_digest(reviews);
    let report_sha256 = binding_report_digest(
        declared_coverage.report_sha256(),
        frozen_policy.policy_sha256(),
        &review_set_sha256,
        closure,
        decision_binding_level,
        claim_coverage_level,
        protocol_local_coverage_established,
        &missing_review_keys,
        &rejected_review_keys,
        &inconclusive_review_keys,
        &invalid_review_keys,
    );

    Ok(EvidenceDecisionBindingReport {
        report_sha256,
        source_coverage_report_sha256: declared_coverage.report_sha256().clone(),
        policy_sha256: frozen_policy.policy_sha256().clone(),
        review_set_sha256,
        closure,
        decision_binding_level,
        claim_coverage_level,
        protocol_local_coverage_established,
        global_exhaustiveness_established: false,
        missing_review_keys,
        rejected_review_keys,
        inconclusive_review_keys,
        invalid_review_keys,
        qualification_established: false,
    })
}

fn decision_class(decision: &CorpusDecisionKind) -> Option<CorpusDecisionClass> {
    match decision {
        CorpusDecisionKind::Include { .. } => Some(CorpusDecisionClass::Include),
        CorpusDecisionKind::Exclude => Some(CorpusDecisionClass::Exclude),
        CorpusDecisionKind::Duplicate { .. } => Some(CorpusDecisionClass::Duplicate),
        CorpusDecisionKind::Unresolved => None,
    }
}

fn corpus_decision_digest(decision: &CorpusItemDecision) -> Sha256Digest {
    let mut digest = FramedDigest::new(DECISION_DIGEST_DOMAIN);
    digest.text(decision.source_id.as_str());
    digest.text(decision.item_sha256.as_str());
    digest.text(decision.decision_policy_sha256.as_str());
    digest.text(decision.rationale_sha256.as_str());
    match &decision.decision {
        CorpusDecisionKind::Include {
            evidence_id,
            evidence_artifact_sha256,
        } => {
            digest.text("include");
            digest.text(evidence_id.as_str());
            digest.text(evidence_artifact_sha256.as_str());
        }
        CorpusDecisionKind::Exclude => digest.text("exclude"),
        CorpusDecisionKind::Duplicate {
            canonical_item_sha256,
        } => {
            digest.text("duplicate");
            digest.text(canonical_item_sha256.as_str());
        }
        CorpusDecisionKind::Unresolved => digest.text("unresolved"),
    }
    digest.finish()
}

fn policy_digest(policy: &EvidenceDecisionBindingPolicy) -> Sha256Digest {
    let mut digest = FramedDigest::new(DECISION_BINDING_POLICY_DOMAIN);
    digest.text(EVIDENCE_DECISION_BINDING_SCHEMA);
    digest.text(policy.policy_id.as_str());
    digest.text(policy.coverage_report_sha256.as_str());
    let mut requirements = policy.requirements.clone();
    requirements.sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
    for requirement in requirements {
        digest.text("requirement");
        digest.text(requirement.requirement_id.as_str());
        digest.text(decision_class_tag(requirement.decision_class));
        digest.text(review_dimension_tag(requirement.dimension));
        digest.text(requirement.reviewer_identity_sha256.as_str());
        digest.text(requirement.review_protocol_sha256.as_str());
    }
    digest.optional_sha(policy.supersedes_policy_sha256.as_ref());
    digest.finish()
}

fn review_set_digest(reviews: &[CorpusDecisionSemanticReview]) -> Sha256Digest {
    let mut ordered = reviews.to_vec();
    ordered.sort_by(|left, right| {
        (&left.source_id, &left.item_sha256, &left.requirement_id)
            .cmp(&(&right.source_id, &right.item_sha256, &right.requirement_id))
    });
    let mut digest = FramedDigest::new(DECISION_REVIEW_SET_DOMAIN);
    for review in ordered {
        digest.text("review");
        digest.text(review.requirement_id.as_str());
        digest.text(review.source_id.as_str());
        digest.text(review.item_sha256.as_str());
        digest.text(review.decision_sha256.as_str());
        digest.text(decision_class_tag(review.decision_class));
        digest.text(review_dimension_tag(review.dimension));
        digest.text(review.reviewer_identity_sha256.as_str());
        digest.text(review.review_protocol_sha256.as_str());
        digest.text(review.review_artifact_sha256.as_str());
        digest.text(review_outcome_tag(review.outcome));
    }
    digest.finish()
}

#[allow(clippy::too_many_arguments)]
fn binding_report_digest(
    source_coverage_report_sha256: &Sha256Digest,
    policy_sha256: &Sha256Digest,
    review_set_sha256: &Sha256Digest,
    closure: EvidenceDecisionBindingClosure,
    decision_binding_level: AuthorityLevel,
    claim_coverage_level: AuthorityLevel,
    local_coverage_established: bool,
    missing: &BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
    rejected: &BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
    inconclusive: &BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
    invalid: &BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(DECISION_BINDING_REPORT_DOMAIN);
    digest.text(source_coverage_report_sha256.as_str());
    digest.text(policy_sha256.as_str());
    digest.text(review_set_sha256.as_str());
    digest.text(binding_closure_tag(closure));
    digest.text(authority_level_tag(decision_binding_level));
    digest.text(authority_level_tag(claim_coverage_level));
    digest.text(if local_coverage_established {
        "protocol-local-coverage-established"
    } else {
        "protocol-local-coverage-not-established"
    });
    digest.text("global-exhaustiveness-not-established");
    digest.text("qualification-not-established");
    digest_review_keys(&mut digest, "missing", missing);
    digest_review_keys(&mut digest, "rejected", rejected);
    digest_review_keys(&mut digest, "inconclusive", inconclusive);
    digest_review_keys(&mut digest, "invalid", invalid);
    digest.finish()
}

fn digest_review_keys(
    digest: &mut FramedDigest,
    tag: &str,
    keys: &BTreeSet<(ResearchId, Sha256Digest, ResearchId)>,
) {
    for (source_id, item_sha256, requirement_id) in keys {
        digest.text(tag);
        digest.text(source_id.as_str());
        digest.text(item_sha256.as_str());
        digest.text(requirement_id.as_str());
    }
}

const fn decision_class_tag(class: CorpusDecisionClass) -> &'static str {
    match class {
        CorpusDecisionClass::Include => "include",
        CorpusDecisionClass::Exclude => "exclude",
        CorpusDecisionClass::Duplicate => "duplicate",
    }
}

const fn review_dimension_tag(dimension: CorpusDecisionReviewDimension) -> &'static str {
    match dimension {
        CorpusDecisionReviewDimension::ClaimRelevance => "claim-relevance",
        CorpusDecisionReviewDimension::InclusionEligibility => "inclusion-eligibility",
        CorpusDecisionReviewDimension::ExclusionJustification => "exclusion-justification",
        CorpusDecisionReviewDimension::DuplicateIdentity => "duplicate-identity",
        CorpusDecisionReviewDimension::ScopeCompatibility => "scope-compatibility",
        CorpusDecisionReviewDimension::Other => "other",
    }
}

const fn review_outcome_tag(outcome: SemanticReviewOutcome) -> &'static str {
    match outcome {
        SemanticReviewOutcome::Accepted => "accepted",
        SemanticReviewOutcome::Rejected => "rejected",
        SemanticReviewOutcome::Inconclusive => "inconclusive",
        SemanticReviewOutcome::Invalid => "invalid",
    }
}

const fn binding_closure_tag(closure: EvidenceDecisionBindingClosure) -> &'static str {
    match closure {
        EvidenceDecisionBindingClosure::Bound => "bound",
        EvidenceDecisionBindingClosure::Incomplete => "incomplete",
        EvidenceDecisionBindingClosure::Rejected => "rejected",
        EvidenceDecisionBindingClosure::Invalid => "invalid",
    }
}

const fn authority_level_tag(level: AuthorityLevel) -> &'static str {
    match level {
        AuthorityLevel::None => "none",
        AuthorityLevel::Declared => "declared",
        AuthorityLevel::Bound => "bound",
    }
}

struct FramedDigest {
    bytes: Vec<u8>,
}

impl FramedDigest {
    fn new(domain: &str) -> Self {
        let mut digest = Self { bytes: Vec::new() };
        digest.text(domain);
        digest
    }

    fn text(&mut self, value: &str) {
        self.bytes
            .extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn optional_sha(&mut self, value: Option<&Sha256Digest>) {
        match value {
            Some(value) => {
                self.text("some");
                self.text(value.as_str());
            }
            None => self.text("none"),
        }
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_decisions_are_not_reviewable_classes() {
        assert_eq!(decision_class(&CorpusDecisionKind::Unresolved), None);
    }

    #[test]
    fn decision_digest_is_sensitive_to_rationale() {
        let mut decision = CorpusItemDecision {
            source_id: ResearchId::parse("SOURCE-1").unwrap(),
            item_sha256: Sha256Digest::of_bytes(b"item"),
            decision_policy_sha256: Sha256Digest::of_bytes(b"policy"),
            rationale_sha256: Sha256Digest::of_bytes(b"rationale-a"),
            decision: CorpusDecisionKind::Exclude,
        };
        let first = corpus_decision_digest(&decision);
        decision.rationale_sha256 = Sha256Digest::of_bytes(b"rationale-b");
        assert_ne!(first, corpus_decision_digest(&decision));
    }
}
