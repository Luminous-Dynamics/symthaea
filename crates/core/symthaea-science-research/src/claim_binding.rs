// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Semantic binding for evidence-to-claim relations.
//!
//! SCI-009A deliberately caps caller-declared claim/evidence relations at
//! `Declared`. This layer can raise the *relation binding* to `Bound`, but only
//! after rerunning the exact declared adjudication and satisfying an exact,
//! frozen semantic-review policy. Bound here still does not mean qualified:
//! reviewer identity, protocol, and review artifacts are retained as ordinary
//! evidence commitments for a later qualification capability to authenticate.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    AdjudicatedScientificClaim, AuthorityFacet, AuthorityLevel, AuthorityProfile,
    ClaimEvidenceLink, ClaimEvidenceRelation, EvidenceRecord, ResearchId, ScientificClaim,
    Sha256Digest, adjudicate_scientific_claim,
};

pub const CLAIM_RELATION_BINDING_SCHEMA: &str = "symthaea.claim-relation-binding.v1";
const BINDING_POLICY_DIGEST_DOMAIN: &str = "symthaea.claim-relation-binding-policy.identity.v1";
const BINDING_REPORT_DIGEST_DOMAIN: &str = "symthaea.claim-relation-binding-report.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SemanticReviewDimension {
    SubjectIdentity,
    EvidenceApplicability,
    Directionality,
    ScopeCompatibility,
    MeasurementSemantics,
    StatisticalInterpretation,
    CausalInterpretation,
    FormalizationEquivalence,
    DomainInterpretation,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationReviewRequirement {
    pub requirement_id: ResearchId,
    pub relation: ClaimEvidenceRelation,
    pub dimension: SemanticReviewDimension,
    pub reviewer_identity_sha256: Sha256Digest,
    pub review_protocol_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimRelationBindingPolicy {
    pub schema_version: String,
    pub policy_id: ResearchId,
    pub adjudication_sha256: Sha256Digest,
    pub requirements: Vec<RelationReviewRequirement>,
    /// Lineage only; supersession does not transfer semantic authority.
    pub supersedes_policy_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimRelationBindingPolicyIssue {
    WrongSchemaVersion { found: String },
    MissingRequirements,
    DuplicateRequirementId { requirement_id: ResearchId },
    NonBindableRelationRequirement {
        requirement_id: ResearchId,
        relation: ClaimEvidenceRelation,
    },
}

impl ClaimRelationBindingPolicy {
    pub fn validate(&self) -> Vec<ClaimRelationBindingPolicyIssue> {
        let mut issues = Vec::new();
        if self.schema_version != CLAIM_RELATION_BINDING_SCHEMA {
            issues.push(ClaimRelationBindingPolicyIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.requirements.is_empty() {
            issues.push(ClaimRelationBindingPolicyIssue::MissingRequirements);
        }
        let mut requirement_ids = BTreeSet::new();
        for requirement in &self.requirements {
            if !requirement_ids.insert(requirement.requirement_id.clone()) {
                issues.push(ClaimRelationBindingPolicyIssue::DuplicateRequirementId {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }
            if !is_bindable_relation(requirement.relation) {
                issues.push(ClaimRelationBindingPolicyIssue::NonBindableRelationRequirement {
                    requirement_id: requirement.requirement_id.clone(),
                    relation: requirement.relation,
                });
            }
        }
        issues
    }

    pub fn freeze(
        self,
    ) -> Result<FrozenClaimRelationBindingPolicy, Vec<ClaimRelationBindingPolicyIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let policy_sha256 = policy_digest(&self);
        Ok(FrozenClaimRelationBindingPolicy {
            policy: self,
            policy_sha256,
        })
    }
}

/// Frozen policy must be rebuilt from its draft and validated on import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenClaimRelationBindingPolicy {
    policy: ClaimRelationBindingPolicy,
    policy_sha256: Sha256Digest,
}

impl FrozenClaimRelationBindingPolicy {
    pub fn policy(&self) -> &ClaimRelationBindingPolicy {
        &self.policy
    }

    pub fn policy_sha256(&self) -> &Sha256Digest {
        &self.policy_sha256
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SemanticReviewOutcome {
    Accepted,
    Rejected,
    Inconclusive,
    Invalid,
}

/// Exact review evidence for one relation requirement on one evidence link.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationSemanticReview {
    pub requirement_id: ResearchId,
    pub evidence_id: ResearchId,
    pub evidence_artifact_sha256: Sha256Digest,
    pub relation: ClaimEvidenceRelation,
    pub relation_rationale_sha256: Sha256Digest,
    pub dimension: SemanticReviewDimension,
    pub reviewer_identity_sha256: Sha256Digest,
    pub review_protocol_sha256: Sha256Digest,
    pub review_artifact_sha256: Sha256Digest,
    pub outcome: SemanticReviewOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimRelationBindingIssue {
    DeclaredAdjudicationCouldNotBeReplayed,
    DeclaredAdjudicationMismatch,
    PolicyAdjudicationMismatch,
    MissingRequirementForRelation { relation: ClaimEvidenceRelation },
    DuplicateReview {
        evidence_id: ResearchId,
        requirement_id: ResearchId,
    },
    UnknownRequirement { requirement_id: ResearchId },
    UnknownLinkedEvidence { evidence_id: ResearchId },
    RequirementRelationMismatch { evidence_id: ResearchId },
    ReviewEvidenceArtifactMismatch { evidence_id: ResearchId },
    ReviewRelationMismatch { evidence_id: ResearchId },
    ReviewRationaleMismatch { evidence_id: ResearchId },
    ReviewDimensionMismatch { evidence_id: ResearchId },
    ReviewerIdentityMismatch { evidence_id: ResearchId },
    ReviewProtocolMismatch { evidence_id: ResearchId },
}

/// Relation-review closure. `Bound` means every claim-bearing relation satisfied
/// every exact review requirement. It does not mean the claim is true or
/// scientifically qualified.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum RelationBindingClosure {
    Bound,
    Incomplete,
    Rejected,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClaimRelationBindingReport {
    binding_sha256: Sha256Digest,
    adjudication_sha256: Sha256Digest,
    policy_sha256: Sha256Digest,
    review_set_sha256: Sha256Digest,
    closure: RelationBindingClosure,
    relation_binding_level: AuthorityLevel,
    reviewed_evidence_ids: BTreeSet<ResearchId>,
    missing_review_pairs: BTreeSet<(ResearchId, ResearchId)>,
    rejected_review_pairs: BTreeSet<(ResearchId, ResearchId)>,
    inconclusive_review_pairs: BTreeSet<(ResearchId, ResearchId)>,
    invalid_review_pairs: BTreeSet<(ResearchId, ResearchId)>,
    /// Authority already safe under caller-declared relations from SCI-009A.
    declared_claimable_authority: AuthorityProfile,
    /// Authority safe to expose only when semantic relation review is fully
    /// bound and no contradiction/unresolved evidence remains.
    bound_claimable_authority: AuthorityProfile,
    /// Qualification remains a distinct future capability.
    qualification_established: bool,
}

impl ClaimRelationBindingReport {
    pub fn binding_sha256(&self) -> &Sha256Digest {
        &self.binding_sha256
    }

    pub fn adjudication_sha256(&self) -> &Sha256Digest {
        &self.adjudication_sha256
    }

    pub fn policy_sha256(&self) -> &Sha256Digest {
        &self.policy_sha256
    }

    pub fn review_set_sha256(&self) -> &Sha256Digest {
        &self.review_set_sha256
    }

    pub fn closure(&self) -> RelationBindingClosure {
        self.closure
    }

    pub fn relation_binding_level(&self) -> AuthorityLevel {
        self.relation_binding_level
    }

    pub fn declared_claimable_authority(&self) -> &AuthorityProfile {
        &self.declared_claimable_authority
    }

    pub fn bound_claimable_authority(&self) -> &AuthorityProfile {
        &self.bound_claimable_authority
    }

    pub fn qualification_established(&self) -> bool {
        self.qualification_established
    }
}

pub fn bind_claim_relations(
    source_claim: &ScientificClaim,
    evidence: &[EvidenceRecord],
    links: &[ClaimEvidenceLink],
    declared: &AdjudicatedScientificClaim,
    frozen_policy: &FrozenClaimRelationBindingPolicy,
    reviews: &[RelationSemanticReview],
) -> Result<ClaimRelationBindingReport, Vec<ClaimRelationBindingIssue>> {
    let replayed = adjudicate_scientific_claim(source_claim, evidence, links)
        .map_err(|_| vec![ClaimRelationBindingIssue::DeclaredAdjudicationCouldNotBeReplayed])?;
    if replayed.adjudication_sha256() != declared.adjudication_sha256() {
        return Err(vec![ClaimRelationBindingIssue::DeclaredAdjudicationMismatch]);
    }

    let policy = frozen_policy.policy();
    if &policy.adjudication_sha256 != declared.adjudication_sha256() {
        return Err(vec![ClaimRelationBindingIssue::PolicyAdjudicationMismatch]);
    }

    let links_by_id: BTreeMap<_, _> = links
        .iter()
        .map(|link| (link.evidence_id.clone(), link))
        .collect();
    let requirements_by_id: BTreeMap<_, _> = policy
        .requirements
        .iter()
        .map(|requirement| (requirement.requirement_id.clone(), requirement))
        .collect();

    let present_bindable_relations: BTreeSet<_> = links
        .iter()
        .map(|link| link.relation)
        .filter(|relation| is_bindable_relation(*relation))
        .collect();
    let policy_relations: BTreeSet<_> = policy.requirements.iter().map(|item| item.relation).collect();

    let mut issues = Vec::new();
    for relation in &present_bindable_relations {
        if !policy_relations.contains(relation) {
            issues.push(ClaimRelationBindingIssue::MissingRequirementForRelation {
                relation: *relation,
            });
        }
    }

    let mut reviews_by_pair = BTreeMap::new();
    for review in reviews {
        let key = (review.evidence_id.clone(), review.requirement_id.clone());
        if reviews_by_pair.insert(key.clone(), review).is_some() {
            issues.push(ClaimRelationBindingIssue::DuplicateReview {
                evidence_id: key.0,
                requirement_id: key.1,
            });
            continue;
        }
        let Some(requirement) = requirements_by_id.get(&review.requirement_id) else {
            issues.push(ClaimRelationBindingIssue::UnknownRequirement {
                requirement_id: review.requirement_id.clone(),
            });
            continue;
        };
        let Some(link) = links_by_id.get(&review.evidence_id) else {
            issues.push(ClaimRelationBindingIssue::UnknownLinkedEvidence {
                evidence_id: review.evidence_id.clone(),
            });
            continue;
        };
        if requirement.relation != link.relation {
            issues.push(ClaimRelationBindingIssue::RequirementRelationMismatch {
                evidence_id: review.evidence_id.clone(),
            });
        }
        if review.evidence_artifact_sha256 != link.evidence_artifact_sha256 {
            issues.push(ClaimRelationBindingIssue::ReviewEvidenceArtifactMismatch {
                evidence_id: review.evidence_id.clone(),
            });
        }
        if review.relation != link.relation {
            issues.push(ClaimRelationBindingIssue::ReviewRelationMismatch {
                evidence_id: review.evidence_id.clone(),
            });
        }
        if review.relation_rationale_sha256 != link.relation_rationale_sha256 {
            issues.push(ClaimRelationBindingIssue::ReviewRationaleMismatch {
                evidence_id: review.evidence_id.clone(),
            });
        }
        if review.dimension != requirement.dimension {
            issues.push(ClaimRelationBindingIssue::ReviewDimensionMismatch {
                evidence_id: review.evidence_id.clone(),
            });
        }
        if review.reviewer_identity_sha256 != requirement.reviewer_identity_sha256 {
            issues.push(ClaimRelationBindingIssue::ReviewerIdentityMismatch {
                evidence_id: review.evidence_id.clone(),
            });
        }
        if review.review_protocol_sha256 != requirement.review_protocol_sha256 {
            issues.push(ClaimRelationBindingIssue::ReviewProtocolMismatch {
                evidence_id: review.evidence_id.clone(),
            });
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let mut required_pairs = BTreeSet::new();
    for link in links.iter().filter(|link| is_bindable_relation(link.relation)) {
        for requirement in policy
            .requirements
            .iter()
            .filter(|item| item.relation == link.relation)
        {
            required_pairs.insert((link.evidence_id.clone(), requirement.requirement_id.clone()));
        }
    }

    let mut missing_review_pairs = BTreeSet::new();
    let mut rejected_review_pairs = BTreeSet::new();
    let mut inconclusive_review_pairs = BTreeSet::new();
    let mut invalid_review_pairs = BTreeSet::new();
    let mut reviewed_evidence_ids = BTreeSet::new();

    for pair in &required_pairs {
        let Some(review) = reviews_by_pair.get(pair) else {
            missing_review_pairs.insert(pair.clone());
            continue;
        };
        reviewed_evidence_ids.insert(pair.0.clone());
        match review.outcome {
            SemanticReviewOutcome::Accepted => {}
            SemanticReviewOutcome::Rejected => {
                rejected_review_pairs.insert(pair.clone());
            }
            SemanticReviewOutcome::Inconclusive => {
                inconclusive_review_pairs.insert(pair.clone());
            }
            SemanticReviewOutcome::Invalid => {
                invalid_review_pairs.insert(pair.clone());
            }
        }
    }

    let has_unresolved = !declared.unresolved_evidence_ids().is_empty();
    let no_claim_bearing_relations = required_pairs.is_empty();
    let closure = if !invalid_review_pairs.is_empty() {
        RelationBindingClosure::Invalid
    } else if !rejected_review_pairs.is_empty() {
        RelationBindingClosure::Rejected
    } else if !missing_review_pairs.is_empty()
        || !inconclusive_review_pairs.is_empty()
        || has_unresolved
        || no_claim_bearing_relations
    {
        RelationBindingClosure::Incomplete
    } else {
        RelationBindingClosure::Bound
    };

    let relation_binding_level = if closure == RelationBindingClosure::Bound {
        AuthorityLevel::Bound
    } else {
        AuthorityLevel::Declared
    };

    let bound_claimable_authority = if closure == RelationBindingClosure::Bound
        && declared.contradictory_evidence_ids().is_empty()
        && declared.unresolved_evidence_ids().is_empty()
    {
        declared.support_authority().clone().bounded_by(&bound_relation_ceiling())
    } else {
        AuthorityProfile::empty()
    };

    let review_set_sha256 = review_set_digest(reviews);
    let binding_sha256 = binding_report_digest(
        declared.adjudication_sha256(),
        frozen_policy.policy_sha256(),
        &review_set_sha256,
        closure,
        relation_binding_level,
        declared.claimable_authority(),
        &bound_claimable_authority,
        &missing_review_pairs,
        &rejected_review_pairs,
        &inconclusive_review_pairs,
        &invalid_review_pairs,
    );

    Ok(ClaimRelationBindingReport {
        binding_sha256,
        adjudication_sha256: declared.adjudication_sha256().clone(),
        policy_sha256: frozen_policy.policy_sha256().clone(),
        review_set_sha256,
        closure,
        relation_binding_level,
        reviewed_evidence_ids,
        missing_review_pairs,
        rejected_review_pairs,
        inconclusive_review_pairs,
        invalid_review_pairs,
        declared_claimable_authority: declared.claimable_authority().clone(),
        bound_claimable_authority,
        qualification_established: false,
    })
}

const fn is_bindable_relation(relation: ClaimEvidenceRelation) -> bool {
    matches!(
        relation,
        ClaimEvidenceRelation::Supports
            | ClaimEvidenceRelation::Contradicts
            | ClaimEvidenceRelation::LimitsApplicability
    )
}

fn bound_relation_ceiling() -> AuthorityProfile {
    AuthorityFacet::ALL.into_iter().fold(
        AuthorityProfile::empty(),
        |profile, facet| profile.with(facet, AuthorityLevel::Bound),
    )
}

fn policy_digest(policy: &ClaimRelationBindingPolicy) -> Sha256Digest {
    let mut digest = FramedDigest::new(BINDING_POLICY_DIGEST_DOMAIN);
    digest.text(CLAIM_RELATION_BINDING_SCHEMA);
    digest.text(policy.policy_id.as_str());
    digest.text(policy.adjudication_sha256.as_str());
    let mut requirements = policy.requirements.clone();
    requirements.sort_by(|left, right| left.requirement_id.cmp(&right.requirement_id));
    for requirement in requirements {
        digest.text("requirement");
        digest.text(requirement.requirement_id.as_str());
        digest.text(relation_tag(requirement.relation));
        digest.text(review_dimension_tag(requirement.dimension));
        digest.text(requirement.reviewer_identity_sha256.as_str());
        digest.text(requirement.review_protocol_sha256.as_str());
    }
    digest.optional_sha(policy.supersedes_policy_sha256.as_ref());
    digest.finish()
}

fn review_set_digest(reviews: &[RelationSemanticReview]) -> Sha256Digest {
    let mut ordered = reviews.to_vec();
    ordered.sort_by(|left, right| {
        (&left.evidence_id, &left.requirement_id).cmp(&(&right.evidence_id, &right.requirement_id))
    });
    let mut digest = FramedDigest::new("symthaea.claim-relation-review-set.identity.v1");
    for review in ordered {
        digest.text("review");
        digest.text(review.requirement_id.as_str());
        digest.text(review.evidence_id.as_str());
        digest.text(review.evidence_artifact_sha256.as_str());
        digest.text(relation_tag(review.relation));
        digest.text(review.relation_rationale_sha256.as_str());
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
    adjudication_sha256: &Sha256Digest,
    policy_sha256: &Sha256Digest,
    review_set_sha256: &Sha256Digest,
    closure: RelationBindingClosure,
    relation_binding_level: AuthorityLevel,
    declared_authority: &AuthorityProfile,
    bound_authority: &AuthorityProfile,
    missing: &BTreeSet<(ResearchId, ResearchId)>,
    rejected: &BTreeSet<(ResearchId, ResearchId)>,
    inconclusive: &BTreeSet<(ResearchId, ResearchId)>,
    invalid: &BTreeSet<(ResearchId, ResearchId)>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(BINDING_REPORT_DIGEST_DOMAIN);
    digest.text(adjudication_sha256.as_str());
    digest.text(policy_sha256.as_str());
    digest.text(review_set_sha256.as_str());
    digest.text(binding_closure_tag(closure));
    digest.text(authority_level_tag(relation_binding_level));
    digest_authority(&mut digest, "declared-authority", declared_authority);
    digest_authority(&mut digest, "bound-authority", bound_authority);
    digest_pairs(&mut digest, "missing", missing);
    digest_pairs(&mut digest, "rejected", rejected);
    digest_pairs(&mut digest, "inconclusive", inconclusive);
    digest_pairs(&mut digest, "invalid", invalid);
    digest.finish()
}

fn digest_authority(digest: &mut FramedDigest, tag: &str, authority: &AuthorityProfile) {
    digest.text(tag);
    for facet in AuthorityFacet::ALL {
        digest.text(authority_facet_tag(facet));
        digest.text(authority_level_tag(authority.get(facet)));
    }
}

fn digest_pairs(
    digest: &mut FramedDigest,
    tag: &str,
    pairs: &BTreeSet<(ResearchId, ResearchId)>,
) {
    for (evidence_id, requirement_id) in pairs {
        digest.text(tag);
        digest.text(evidence_id.as_str());
        digest.text(requirement_id.as_str());
    }
}

const fn relation_tag(relation: ClaimEvidenceRelation) -> &'static str {
    match relation {
        ClaimEvidenceRelation::Supports => "supports",
        ClaimEvidenceRelation::Contradicts => "contradicts",
        ClaimEvidenceRelation::LimitsApplicability => "limits-applicability",
        ClaimEvidenceRelation::ContextOnly => "context-only",
        ClaimEvidenceRelation::Unresolved => "unresolved",
    }
}

const fn review_dimension_tag(dimension: SemanticReviewDimension) -> &'static str {
    match dimension {
        SemanticReviewDimension::SubjectIdentity => "subject-identity",
        SemanticReviewDimension::EvidenceApplicability => "evidence-applicability",
        SemanticReviewDimension::Directionality => "directionality",
        SemanticReviewDimension::ScopeCompatibility => "scope-compatibility",
        SemanticReviewDimension::MeasurementSemantics => "measurement-semantics",
        SemanticReviewDimension::StatisticalInterpretation => "statistical-interpretation",
        SemanticReviewDimension::CausalInterpretation => "causal-interpretation",
        SemanticReviewDimension::FormalizationEquivalence => "formalization-equivalence",
        SemanticReviewDimension::DomainInterpretation => "domain-interpretation",
        SemanticReviewDimension::Other => "other",
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

const fn binding_closure_tag(closure: RelationBindingClosure) -> &'static str {
    match closure {
        RelationBindingClosure::Bound => "bound",
        RelationBindingClosure::Incomplete => "incomplete",
        RelationBindingClosure::Rejected => "rejected",
        RelationBindingClosure::Invalid => "invalid",
    }
}

const fn authority_facet_tag(facet: AuthorityFacet) -> &'static str {
    match facet {
        AuthorityFacet::Provenance => "provenance",
        AuthorityFacet::Execution => "execution",
        AuthorityFacet::Empirical => "empirical",
        AuthorityFacet::Causal => "causal",
        AuthorityFacet::Formal => "formal",
        AuthorityFacet::Replication => "replication",
        AuthorityFacet::Independence => "independence",
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
        self.bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
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
    use crate::{EvidenceKind, EvidenceState};

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn computation(subject: &Sha256Digest, evidence_id: &str) -> EvidenceRecord {
        EvidenceRecord::new(
            id(evidence_id),
            subject.clone(),
            EvidenceKind::DeterministicComputation,
            EvidenceState::Pass,
            sha(&format!("artifact-{evidence_id}")),
            [],
            None,
            AuthorityProfile::empty()
                .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
                .with(AuthorityFacet::Execution, AuthorityLevel::Bound),
        )
        .unwrap()
    }

    fn counterexample(subject: &Sha256Digest, evidence_id: &str) -> EvidenceRecord {
        EvidenceRecord::new(
            id(evidence_id),
            subject.clone(),
            EvidenceKind::Counterexample,
            EvidenceState::Pass,
            sha(&format!("artifact-{evidence_id}")),
            [],
            None,
            AuthorityProfile::empty()
                .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
                .with(AuthorityFacet::Execution, AuthorityLevel::Bound)
                .with(AuthorityFacet::Formal, AuthorityLevel::Bound),
        )
        .unwrap()
    }

    fn link(record: &EvidenceRecord, relation: ClaimEvidenceRelation) -> ClaimEvidenceLink {
        ClaimEvidenceLink {
            evidence_id: record.evidence_id().clone(),
            evidence_artifact_sha256: record.artifact_sha256().clone(),
            relation,
            relation_rationale_sha256: sha(&format!("rationale-{}", record.evidence_id().as_str())),
        }
    }

    fn adjudicated(
        evidence: &[EvidenceRecord],
        links: &[ClaimEvidenceLink],
    ) -> (ScientificClaim, AdjudicatedScientificClaim) {
        let subject = evidence[0].subject_sha256().clone();
        let claim = ScientificClaim::from_evidence(id("CLAIM-1"), subject, evidence).unwrap();
        let adjudicated = adjudicate_scientific_claim(&claim, evidence, links).unwrap();
        (claim, adjudicated)
    }

    fn policy(
        adjudicated: &AdjudicatedScientificClaim,
        relation: ClaimEvidenceRelation,
    ) -> FrozenClaimRelationBindingPolicy {
        ClaimRelationBindingPolicy {
            schema_version: CLAIM_RELATION_BINDING_SCHEMA.into(),
            policy_id: id("BIND-POLICY-1"),
            adjudication_sha256: adjudicated.adjudication_sha256().clone(),
            requirements: vec![RelationReviewRequirement {
                requirement_id: id("REVIEW-APPLICABILITY"),
                relation,
                dimension: SemanticReviewDimension::EvidenceApplicability,
                reviewer_identity_sha256: sha("reviewer"),
                review_protocol_sha256: sha("review-protocol"),
            }],
            supersedes_policy_sha256: None,
        }
        .freeze()
        .unwrap()
    }

    fn accepted_review(
        link: &ClaimEvidenceLink,
        outcome: SemanticReviewOutcome,
    ) -> RelationSemanticReview {
        RelationSemanticReview {
            requirement_id: id("REVIEW-APPLICABILITY"),
            evidence_id: link.evidence_id.clone(),
            evidence_artifact_sha256: link.evidence_artifact_sha256.clone(),
            relation: link.relation,
            relation_rationale_sha256: link.relation_rationale_sha256.clone(),
            dimension: SemanticReviewDimension::EvidenceApplicability,
            reviewer_identity_sha256: sha("reviewer"),
            review_protocol_sha256: sha("review-protocol"),
            review_artifact_sha256: sha("review-artifact"),
            outcome,
        }
    }

    #[test]
    fn accepted_exact_review_can_bind_support_authority() {
        let subject = sha("subject");
        let evidence = vec![computation(&subject, "E-1")];
        let links = vec![link(&evidence[0], ClaimEvidenceRelation::Supports)];
        let (claim, adjudicated) = adjudicated(&evidence, &links);
        assert_eq!(
            adjudicated.claimable_authority().get(AuthorityFacet::Execution),
            AuthorityLevel::Declared
        );
        let report = bind_claim_relations(
            &claim,
            &evidence,
            &links,
            &adjudicated,
            &policy(&adjudicated, ClaimEvidenceRelation::Supports),
            &[accepted_review(&links[0], SemanticReviewOutcome::Accepted)],
        )
        .unwrap();
        assert_eq!(report.closure(), RelationBindingClosure::Bound);
        assert_eq!(report.relation_binding_level(), AuthorityLevel::Bound);
        assert_eq!(
            report.bound_claimable_authority().get(AuthorityFacet::Execution),
            AuthorityLevel::Bound
        );
        assert!(!report.qualification_established());
    }

    #[test]
    fn missing_review_never_upgrades_declared_relation() {
        let subject = sha("subject");
        let evidence = vec![computation(&subject, "E-1")];
        let links = vec![link(&evidence[0], ClaimEvidenceRelation::Supports)];
        let (claim, adjudicated) = adjudicated(&evidence, &links);
        let report = bind_claim_relations(
            &claim,
            &evidence,
            &links,
            &adjudicated,
            &policy(&adjudicated, ClaimEvidenceRelation::Supports),
            &[],
        )
        .unwrap();
        assert_eq!(report.closure(), RelationBindingClosure::Incomplete);
        assert_eq!(report.relation_binding_level(), AuthorityLevel::Declared);
        assert!(report.bound_claimable_authority().is_empty());
    }

    #[test]
    fn rejected_semantic_relation_is_not_bound() {
        let subject = sha("subject");
        let evidence = vec![computation(&subject, "E-1")];
        let links = vec![link(&evidence[0], ClaimEvidenceRelation::Supports)];
        let (claim, adjudicated) = adjudicated(&evidence, &links);
        let report = bind_claim_relations(
            &claim,
            &evidence,
            &links,
            &adjudicated,
            &policy(&adjudicated, ClaimEvidenceRelation::Supports),
            &[accepted_review(&links[0], SemanticReviewOutcome::Rejected)],
        )
        .unwrap();
        assert_eq!(report.closure(), RelationBindingClosure::Rejected);
        assert!(report.bound_claimable_authority().is_empty());
    }

    #[test]
    fn accepted_counterexample_relation_never_grants_claim_authority() {
        let subject = sha("subject");
        let evidence = vec![counterexample(&subject, "E-COUNTER")];
        let links = vec![link(&evidence[0], ClaimEvidenceRelation::Contradicts)];
        let (claim, adjudicated) = adjudicated(&evidence, &links);
        let report = bind_claim_relations(
            &claim,
            &evidence,
            &links,
            &adjudicated,
            &policy(&adjudicated, ClaimEvidenceRelation::Contradicts),
            &[accepted_review(&links[0], SemanticReviewOutcome::Accepted)],
        )
        .unwrap();
        assert_eq!(report.closure(), RelationBindingClosure::Bound);
        assert!(report.bound_claimable_authority().is_empty());
    }

    #[test]
    fn unresolved_evidence_blocks_bound_claim_authority() {
        let subject = sha("subject");
        let support = computation(&subject, "E-SUPPORT");
        let unresolved = computation(&subject, "E-UNRESOLVED");
        let evidence = vec![support, unresolved];
        let links = vec![
            link(&evidence[0], ClaimEvidenceRelation::Supports),
            link(&evidence[1], ClaimEvidenceRelation::Unresolved),
        ];
        let (claim, adjudicated) = adjudicated(&evidence, &links);
        let report = bind_claim_relations(
            &claim,
            &evidence,
            &links,
            &adjudicated,
            &policy(&adjudicated, ClaimEvidenceRelation::Supports),
            &[accepted_review(&links[0], SemanticReviewOutcome::Accepted)],
        )
        .unwrap();
        assert_eq!(report.closure(), RelationBindingClosure::Incomplete);
        assert!(report.bound_claimable_authority().is_empty());
    }

    #[test]
    fn reviewer_identity_substitution_is_rejected_structurally() {
        let subject = sha("subject");
        let evidence = vec![computation(&subject, "E-1")];
        let links = vec![link(&evidence[0], ClaimEvidenceRelation::Supports)];
        let (claim, adjudicated) = adjudicated(&evidence, &links);
        let mut review = accepted_review(&links[0], SemanticReviewOutcome::Accepted);
        review.reviewer_identity_sha256 = sha("different-reviewer");
        assert!(matches!(
            bind_claim_relations(
                &claim,
                &evidence,
                &links,
                &adjudicated,
                &policy(&adjudicated, ClaimEvidenceRelation::Supports),
                &[review],
            ),
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                ClaimRelationBindingIssue::ReviewerIdentityMismatch { .. }
            ))
        ));
    }
}
