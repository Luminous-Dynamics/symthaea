// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Polarity-aware adjudication for bounded scientific claims.
//!
//! `ScientificClaim::from_evidence` deliberately performs only structural
//! aggregation. This layer adds the missing semantic relation between each exact
//! evidence record and the claim: support, contradiction, applicability limit,
//! context, or unresolved interpretation. Contradictory evidence can therefore
//! never silently increase claimable authority merely because it is itself
//! high-quality evidence.
//!
//! `ClaimEvidenceLink` is caller-constructible, so its semantic relation is only
//! **declared** in this layer. Bound evidence plus an unreviewed relation cannot
//! mint Bound claim authority: exposed claim authority is capped at `Declared`
//! until a later semantic-review capability authenticates the relation itself.
//!
//! Adjudication remains scoped to the exact evidence-ID set already bound by the
//! source `ScientificClaim` plus the exact evidence records re-presented here. It
//! does not claim that this set is exhaustive with respect to all evidence that
//! may exist elsewhere.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    AuthorityFacet, AuthorityLevel, AuthorityProfile, EvidenceKind, EvidenceRecord, EvidenceState,
    ResearchId, ScientificClaim, Sha256Digest,
};

const CLAIM_ADJUDICATION_DIGEST_DOMAIN: &str = "symthaea.claim-adjudication.identity.v1";
const CLAIM_SNAPSHOT_DIGEST_DOMAIN: &str = "symthaea.claim-adjudication.claim-snapshot.v1";
const EVIDENCE_SET_DIGEST_DOMAIN: &str = "symthaea.claim-adjudication.evidence-set.v1";
const RELATION_SET_DIGEST_DOMAIN: &str = "symthaea.claim-adjudication.relation-set.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ClaimEvidenceRelation {
    Supports,
    Contradicts,
    LimitsApplicability,
    ContextOnly,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimEvidenceLink {
    pub evidence_id: ResearchId,
    /// Binds the relation to the exact evidence artifact, not merely a stable ID.
    pub evidence_artifact_sha256: Sha256Digest,
    pub relation: ClaimEvidenceRelation,
    /// Exact rationale/adjudication artifact explaining the relation assignment.
    pub relation_rationale_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimAdjudicationIssue {
    EmptyEvidence,
    InvalidEvidence { evidence_id: ResearchId },
    SubjectMismatch { evidence_id: ResearchId },
    EvidenceSetMismatch,
    DuplicateEvidenceId { evidence_id: ResearchId },
    DuplicateLink { evidence_id: ResearchId },
    MissingLink { evidence_id: ResearchId },
    UnknownLinkedEvidence { evidence_id: ResearchId },
    EvidenceArtifactMismatch { evidence_id: ResearchId },
    PositiveRelationRequiresPassingEvidence { evidence_id: ResearchId },
    CounterexampleMisclassified { evidence_id: ResearchId },
    SupportAuthorityExceedsSourceClaim,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ClaimDisposition {
    Unsupported,
    Unresolved,
    Supported,
    SupportedWithDeclaredLimits,
    Contested,
    RefutedByCounterexample,
}

/// Polarity-aware view of one already-bounded `ScientificClaim`.
///
/// This type is intentionally serializable but not deserializable. Recovery
/// must re-present the source claim, exact evidence records, and exact relation
/// links and rerun `adjudicate_scientific_claim`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdjudicatedScientificClaim {
    adjudication_sha256: Sha256Digest,
    source_claim_sha256: Sha256Digest,
    evidence_set_sha256: Sha256Digest,
    relation_set_sha256: Sha256Digest,
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    supportive_evidence_ids: BTreeSet<ResearchId>,
    contradictory_evidence_ids: BTreeSet<ResearchId>,
    applicability_limit_evidence_ids: BTreeSet<ResearchId>,
    contextual_evidence_ids: BTreeSet<ResearchId>,
    unresolved_evidence_ids: BTreeSet<ResearchId>,
    support_authority: AuthorityProfile,
    claimable_authority: AuthorityProfile,
    relation_binding_level: AuthorityLevel,
    disposition: ClaimDisposition,
}

impl AdjudicatedScientificClaim {
    pub fn adjudication_sha256(&self) -> &Sha256Digest {
        &self.adjudication_sha256
    }

    pub fn source_claim_sha256(&self) -> &Sha256Digest {
        &self.source_claim_sha256
    }

    pub fn evidence_set_sha256(&self) -> &Sha256Digest {
        &self.evidence_set_sha256
    }

    pub fn relation_set_sha256(&self) -> &Sha256Digest {
        &self.relation_set_sha256
    }

    pub fn claim_id(&self) -> &ResearchId {
        &self.claim_id
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }

    pub fn supportive_evidence_ids(&self) -> &BTreeSet<ResearchId> {
        &self.supportive_evidence_ids
    }

    pub fn contradictory_evidence_ids(&self) -> &BTreeSet<ResearchId> {
        &self.contradictory_evidence_ids
    }

    pub fn applicability_limit_evidence_ids(&self) -> &BTreeSet<ResearchId> {
        &self.applicability_limit_evidence_ids
    }

    pub fn contextual_evidence_ids(&self) -> &BTreeSet<ResearchId> {
        &self.contextual_evidence_ids
    }

    pub fn unresolved_evidence_ids(&self) -> &BTreeSet<ResearchId> {
        &self.unresolved_evidence_ids
    }

    /// Authority contributed by evidence explicitly adjudicated as support,
    /// before accounting for the fact that relation semantics are unreviewed.
    pub fn support_authority(&self) -> &AuthorityProfile {
        &self.support_authority
    }

    /// Authority safe to expose under caller-declared evidence relations.
    ///
    /// It is empty whenever contradictory evidence is present and otherwise is
    /// capped at `Declared` per facet until a future semantic-binding capability
    /// authenticates the support relation itself.
    pub fn claimable_authority(&self) -> &AuthorityProfile {
        &self.claimable_authority
    }

    /// Relation semantics are caller-declared in SCI-009A and therefore cannot
    /// exceed `Declared` authority.
    pub fn relation_binding_level(&self) -> AuthorityLevel {
        self.relation_binding_level
    }

    pub fn disposition(&self) -> ClaimDisposition {
        self.disposition
    }
}

/// Re-adjudicate the exact evidence-ID set already bound by `source_claim`.
///
/// The evidence relation is a separate semantic commitment from the evidence
/// artifact itself. High-quality counterevidence therefore remains high-quality
/// evidence while reducing, rather than increasing, the authority of the claim
/// it contradicts.
pub fn adjudicate_scientific_claim(
    source_claim: &ScientificClaim,
    evidence: &[EvidenceRecord],
    links: &[ClaimEvidenceLink],
) -> Result<AdjudicatedScientificClaim, Vec<ClaimAdjudicationIssue>> {
    let mut issues = Vec::new();
    if evidence.is_empty() {
        issues.push(ClaimAdjudicationIssue::EmptyEvidence);
    }

    let mut evidence_by_id = BTreeMap::new();
    for record in evidence {
        if evidence_by_id
            .insert(record.evidence_id().clone(), record)
            .is_some()
        {
            issues.push(ClaimAdjudicationIssue::DuplicateEvidenceId {
                evidence_id: record.evidence_id().clone(),
            });
        }
        if !record.validate().is_empty() {
            issues.push(ClaimAdjudicationIssue::InvalidEvidence {
                evidence_id: record.evidence_id().clone(),
            });
        }
        if record.subject_sha256() != source_claim.subject_sha256() {
            issues.push(ClaimAdjudicationIssue::SubjectMismatch {
                evidence_id: record.evidence_id().clone(),
            });
        }
    }

    let supplied_ids: BTreeSet<_> = evidence_by_id.keys().cloned().collect();
    if &supplied_ids != source_claim.evidence_ids() {
        issues.push(ClaimAdjudicationIssue::EvidenceSetMismatch);
    }

    let mut links_by_id = BTreeMap::new();
    for link in links {
        if links_by_id
            .insert(link.evidence_id.clone(), link)
            .is_some()
        {
            issues.push(ClaimAdjudicationIssue::DuplicateLink {
                evidence_id: link.evidence_id.clone(),
            });
            continue;
        }
        let Some(record) = evidence_by_id.get(&link.evidence_id) else {
            issues.push(ClaimAdjudicationIssue::UnknownLinkedEvidence {
                evidence_id: link.evidence_id.clone(),
            });
            continue;
        };
        if &link.evidence_artifact_sha256 != record.artifact_sha256() {
            issues.push(ClaimAdjudicationIssue::EvidenceArtifactMismatch {
                evidence_id: link.evidence_id.clone(),
            });
        }
        validate_relation(record, link, &mut issues);
    }

    for evidence_id in evidence_by_id.keys() {
        if !links_by_id.contains_key(evidence_id) {
            issues.push(ClaimAdjudicationIssue::MissingLink {
                evidence_id: evidence_id.clone(),
            });
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let mut supportive_evidence_ids = BTreeSet::new();
    let mut contradictory_evidence_ids = BTreeSet::new();
    let mut applicability_limit_evidence_ids = BTreeSet::new();
    let mut contextual_evidence_ids = BTreeSet::new();
    let mut unresolved_evidence_ids = BTreeSet::new();
    let mut support_authority = AuthorityProfile::empty();
    let mut counterexample_refutation = false;

    for (evidence_id, record) in &evidence_by_id {
        let link = links_by_id[evidence_id];
        match link.relation {
            ClaimEvidenceRelation::Supports => {
                supportive_evidence_ids.insert(evidence_id.clone());
                support_authority = support_authority.evidence_union(record.authority());
            }
            ClaimEvidenceRelation::Contradicts => {
                contradictory_evidence_ids.insert(evidence_id.clone());
                if record.kind() == EvidenceKind::Counterexample {
                    counterexample_refutation = true;
                }
            }
            ClaimEvidenceRelation::LimitsApplicability => {
                applicability_limit_evidence_ids.insert(evidence_id.clone());
            }
            ClaimEvidenceRelation::ContextOnly => {
                contextual_evidence_ids.insert(evidence_id.clone());
            }
            ClaimEvidenceRelation::Unresolved => {
                unresolved_evidence_ids.insert(evidence_id.clone());
            }
        }
    }

    if !support_authority.is_within(source_claim.structural_authority()) {
        return Err(vec![ClaimAdjudicationIssue::SupportAuthorityExceedsSourceClaim]);
    }

    let disposition = if counterexample_refutation {
        ClaimDisposition::RefutedByCounterexample
    } else if !contradictory_evidence_ids.is_empty() {
        ClaimDisposition::Contested
    } else if !supportive_evidence_ids.is_empty() && !applicability_limit_evidence_ids.is_empty() {
        ClaimDisposition::SupportedWithDeclaredLimits
    } else if !supportive_evidence_ids.is_empty() {
        ClaimDisposition::Supported
    } else if !unresolved_evidence_ids.is_empty() {
        ClaimDisposition::Unresolved
    } else {
        ClaimDisposition::Unsupported
    };

    let relation_binding_level = AuthorityLevel::Declared;
    let claimable_authority = if contradictory_evidence_ids.is_empty() {
        support_authority.bounded_by(&declared_relation_ceiling())
    } else {
        AuthorityProfile::empty()
    };

    let source_claim_sha256 = claim_snapshot_digest(source_claim);
    let evidence_set_sha256 = evidence_set_digest(evidence_by_id.values().copied());
    let relation_set_sha256 = relation_set_digest(links_by_id.values().copied());
    let adjudication_sha256 = adjudication_digest(
        &source_claim_sha256,
        &evidence_set_sha256,
        &relation_set_sha256,
        disposition,
        relation_binding_level,
        &support_authority,
        &claimable_authority,
    );

    Ok(AdjudicatedScientificClaim {
        adjudication_sha256,
        source_claim_sha256,
        evidence_set_sha256,
        relation_set_sha256,
        claim_id: source_claim.claim_id().clone(),
        subject_sha256: source_claim.subject_sha256().clone(),
        supportive_evidence_ids,
        contradictory_evidence_ids,
        applicability_limit_evidence_ids,
        contextual_evidence_ids,
        unresolved_evidence_ids,
        support_authority,
        claimable_authority,
        relation_binding_level,
        disposition,
    })
}

fn validate_relation(
    record: &EvidenceRecord,
    link: &ClaimEvidenceLink,
    issues: &mut Vec<ClaimAdjudicationIssue>,
) {
    let positive_relation = matches!(
        link.relation,
        ClaimEvidenceRelation::Supports
            | ClaimEvidenceRelation::Contradicts
            | ClaimEvidenceRelation::LimitsApplicability
    );
    if positive_relation && record.state() != EvidenceState::Pass {
        issues.push(ClaimAdjudicationIssue::PositiveRelationRequiresPassingEvidence {
            evidence_id: record.evidence_id().clone(),
        });
    }

    if record.state() == EvidenceState::Pass
        && record.kind() == EvidenceKind::Counterexample
        && link.relation != ClaimEvidenceRelation::Contradicts
    {
        issues.push(ClaimAdjudicationIssue::CounterexampleMisclassified {
            evidence_id: record.evidence_id().clone(),
        });
    }
}

fn declared_relation_ceiling() -> AuthorityProfile {
    AuthorityFacet::ALL.into_iter().fold(
        AuthorityProfile::empty(),
        |profile, facet| profile.with(facet, AuthorityLevel::Declared),
    )
}

fn claim_snapshot_digest(claim: &ScientificClaim) -> Sha256Digest {
    let mut digest = FramedDigest::new(CLAIM_SNAPSHOT_DIGEST_DOMAIN);
    digest.text(claim.claim_id().as_str());
    digest.text(claim.subject_sha256().as_str());
    for evidence_id in claim.evidence_ids() {
        digest.text("evidence-id");
        digest.text(evidence_id.as_str());
    }
    digest_authority(&mut digest, claim.structural_authority());
    digest.finish()
}

fn evidence_set_digest<'a>(records: impl IntoIterator<Item = &'a EvidenceRecord>) -> Sha256Digest {
    let mut ordered: Vec<_> = records.into_iter().collect();
    ordered.sort_by(|left, right| left.evidence_id().cmp(right.evidence_id()));
    let mut digest = FramedDigest::new(EVIDENCE_SET_DIGEST_DOMAIN);
    for record in ordered {
        digest.text("evidence");
        digest.text(record.evidence_id().as_str());
        digest.text(record.subject_sha256().as_str());
        digest.text(evidence_kind_tag(record.kind()));
        digest.text(evidence_state_tag(record.state()));
        digest.text(record.artifact_sha256().as_str());
        for root in record.provenance_roots() {
            digest.text("provenance-root");
            digest.text(root.as_str());
        }
        digest.optional_sha(record.qualification_sha256());
        digest_authority(&mut digest, record.authority());
    }
    digest.finish()
}

fn relation_set_digest<'a>(links: impl IntoIterator<Item = &'a ClaimEvidenceLink>) -> Sha256Digest {
    let mut ordered: Vec<_> = links.into_iter().collect();
    ordered.sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));
    let mut digest = FramedDigest::new(RELATION_SET_DIGEST_DOMAIN);
    for link in ordered {
        digest.text("link");
        digest.text(link.evidence_id.as_str());
        digest.text(link.evidence_artifact_sha256.as_str());
        digest.text(relation_tag(link.relation));
        digest.text(link.relation_rationale_sha256.as_str());
    }
    digest.finish()
}

fn adjudication_digest(
    source_claim_sha256: &Sha256Digest,
    evidence_set_sha256: &Sha256Digest,
    relation_set_sha256: &Sha256Digest,
    disposition: ClaimDisposition,
    relation_binding_level: AuthorityLevel,
    support_authority: &AuthorityProfile,
    claimable_authority: &AuthorityProfile,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(CLAIM_ADJUDICATION_DIGEST_DOMAIN);
    digest.text(source_claim_sha256.as_str());
    digest.text(evidence_set_sha256.as_str());
    digest.text(relation_set_sha256.as_str());
    digest.text(disposition_tag(disposition));
    digest.text("relation-binding");
    digest.text(authority_level_tag(relation_binding_level));
    digest.text("support-authority");
    digest_authority(&mut digest, support_authority);
    digest.text("claimable-authority");
    digest_authority(&mut digest, claimable_authority);
    digest.text("scope:supplied-source-claim-evidence-set-only");
    digest.finish()
}

fn digest_authority(digest: &mut FramedDigest, authority: &AuthorityProfile) {
    for facet in AuthorityFacet::ALL {
        digest.text(authority_facet_tag(facet));
        digest.text(authority_level_tag(authority.get(facet)));
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

const fn disposition_tag(disposition: ClaimDisposition) -> &'static str {
    match disposition {
        ClaimDisposition::Unsupported => "unsupported",
        ClaimDisposition::Unresolved => "unresolved",
        ClaimDisposition::Supported => "supported",
        ClaimDisposition::SupportedWithDeclaredLimits => "supported-with-declared-limits",
        ClaimDisposition::Contested => "contested",
        ClaimDisposition::RefutedByCounterexample => "refuted-by-counterexample",
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

const fn evidence_state_tag(state: EvidenceState) -> &'static str {
    match state {
        EvidenceState::Pass => "pass",
        EvidenceState::Fail => "fail",
        EvidenceState::Unknown => "unknown",
        EvidenceState::Incomplete => "incomplete",
        EvidenceState::NotApplicable => "not-applicable",
        EvidenceState::Invalid => "invalid",
    }
}

const fn evidence_kind_tag(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::StructuralDeclaration => "structural-declaration",
        EvidenceKind::RawObservation => "raw-observation",
        EvidenceKind::CalibratedObservation => "calibrated-observation",
        EvidenceKind::DerivedMeasurement => "derived-measurement",
        EvidenceKind::DeterministicComputation => "deterministic-computation",
        EvidenceKind::NumericalSimulation => "numerical-simulation",
        EvidenceKind::SymbolicDerivation => "symbolic-derivation",
        EvidenceKind::BoundedVerification => "bounded-verification",
        EvidenceKind::Counterexample => "counterexample",
        EvidenceKind::FormalProof => "formal-proof",
        EvidenceKind::FormalProofRejection => "formal-proof-rejection",
        EvidenceKind::CausalStudy => "causal-study",
        EvidenceKind::ReplicationStudy => "replication-study",
        EvidenceKind::LiteratureEvidence => "literature-evidence",
        EvidenceKind::HumanReview => "human-review",
        EvidenceKind::FormalizationReview => "formalization-review",
        EvidenceKind::NegativeControl => "negative-control",
        EvidenceKind::Ablation => "ablation",
        EvidenceKind::RobustnessTest => "robustness-test",
        EvidenceKind::OutOfDistributionTest => "out-of-distribution-test",
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

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn computation_with_authority(
        subject: &Sha256Digest,
        evidence_id: &str,
        authority: AuthorityProfile,
    ) -> EvidenceRecord {
        EvidenceRecord::new(
            id(evidence_id),
            subject.clone(),
            EvidenceKind::DeterministicComputation,
            EvidenceState::Pass,
            sha(&format!("artifact-{evidence_id}")),
            [],
            None,
            authority,
        )
        .unwrap()
    }

    fn computation(subject: &Sha256Digest, evidence_id: &str) -> EvidenceRecord {
        computation_with_authority(
            subject,
            evidence_id,
            AuthorityProfile::empty()
                .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
                .with(AuthorityFacet::Execution, AuthorityLevel::Bound),
        )
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
            relation_rationale_sha256: sha(&format!("rationale-{}", record.evidence_id())),
        }
    }

    #[test]
    fn counterexample_cannot_increase_claimable_authority() {
        let subject = sha("subject");
        let support = computation(&subject, "SUPPORT");
        let refutation = counterexample(&subject, "COUNTEREXAMPLE");
        let evidence = vec![support.clone(), refutation.clone()];
        let source = ScientificClaim::from_evidence(id("CLAIM"), subject, &evidence).unwrap();

        assert_eq!(
            source.structural_authority().get(AuthorityFacet::Formal),
            AuthorityLevel::Bound
        );

        let adjudicated = adjudicate_scientific_claim(
            &source,
            &evidence,
            &[
                link(&support, ClaimEvidenceRelation::Supports),
                link(&refutation, ClaimEvidenceRelation::Contradicts),
            ],
        )
        .unwrap();

        assert_eq!(
            adjudicated.disposition(),
            ClaimDisposition::RefutedByCounterexample
        );
        assert!(adjudicated.claimable_authority().is_empty());
        assert_eq!(
            adjudicated.support_authority().get(AuthorityFacet::Execution),
            AuthorityLevel::Bound
        );
    }

    #[test]
    fn counterexample_misclassified_as_support_fails_closed() {
        let subject = sha("subject");
        let refutation = counterexample(&subject, "COUNTEREXAMPLE");
        let evidence = vec![refutation.clone()];
        let source = ScientificClaim::from_evidence(id("CLAIM"), subject, &evidence).unwrap();
        let result = adjudicate_scientific_claim(
            &source,
            &evidence,
            &[link(&refutation, ClaimEvidenceRelation::Supports)],
        );
        assert!(matches!(
            result,
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                ClaimAdjudicationIssue::CounterexampleMisclassified { .. }
            ))
        ));
    }

    #[test]
    fn omitted_source_evidence_fails_closed() {
        let subject = sha("subject");
        let first = computation(&subject, "A");
        let second = computation(&subject, "B");
        let all = vec![first.clone(), second];
        let source = ScientificClaim::from_evidence(id("CLAIM"), subject, &all).unwrap();
        let result = adjudicate_scientific_claim(
            &source,
            &[first.clone()],
            &[link(&first, ClaimEvidenceRelation::Supports)],
        );
        assert!(matches!(
            result,
            Err(issues) if issues.contains(&ClaimAdjudicationIssue::EvidenceSetMismatch)
        ));
    }

    #[test]
    fn substituted_stronger_record_cannot_escalate_source_claim() {
        let subject = sha("subject");
        let original = computation_with_authority(
            &subject,
            "A",
            AuthorityProfile::empty().with(AuthorityFacet::Provenance, AuthorityLevel::Bound),
        );
        let source = ScientificClaim::from_evidence(id("CLAIM"), subject.clone(), &[original])
            .unwrap();
        let substituted = computation(&subject, "A");
        let result = adjudicate_scientific_claim(
            &source,
            &[substituted.clone()],
            &[link(&substituted, ClaimEvidenceRelation::Supports)],
        );
        assert_eq!(
            result,
            Err(vec![ClaimAdjudicationIssue::SupportAuthorityExceedsSourceClaim])
        );
    }

    #[test]
    fn declared_relation_cannot_mint_bound_claim_authority() {
        let subject = sha("subject");
        let support = computation(&subject, "A");
        let evidence = vec![support.clone()];
        let source = ScientificClaim::from_evidence(id("CLAIM"), subject, &evidence).unwrap();
        let adjudicated = adjudicate_scientific_claim(
            &source,
            &evidence,
            &[link(&support, ClaimEvidenceRelation::Supports)],
        )
        .unwrap();
        assert_eq!(adjudicated.disposition(), ClaimDisposition::Supported);
        assert_eq!(
            adjudicated.support_authority().get(AuthorityFacet::Execution),
            AuthorityLevel::Bound
        );
        assert_eq!(
            adjudicated.claimable_authority().get(AuthorityFacet::Execution),
            AuthorityLevel::Declared
        );
        assert_eq!(adjudicated.relation_binding_level(), AuthorityLevel::Declared);
    }

    #[test]
    fn adjudication_identity_is_order_independent() {
        let subject = sha("subject");
        let a = computation(&subject, "A");
        let b = computation(&subject, "B");
        let evidence = vec![a.clone(), b.clone()];
        let source = ScientificClaim::from_evidence(id("CLAIM"), subject, &evidence).unwrap();
        let first = adjudicate_scientific_claim(
            &source,
            &evidence,
            &[
                link(&a, ClaimEvidenceRelation::Supports),
                link(&b, ClaimEvidenceRelation::ContextOnly),
            ],
        )
        .unwrap();
        let second = adjudicate_scientific_claim(
            &source,
            &[b.clone(), a.clone()],
            &[
                link(&b, ClaimEvidenceRelation::ContextOnly),
                link(&a, ClaimEvidenceRelation::Supports),
            ],
        )
        .unwrap();
        assert_eq!(first.adjudication_sha256(), second.adjudication_sha256());
    }
}
